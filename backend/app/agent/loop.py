"""Tool-calling loop. Phase 1: the LLM gathers data through tools (capped iterations, every call
logged with latency). Phase 2: code computes the safety alerts from the real forecast, then the LLM
writes the structured answer, which is validated (schema, citations, numbers) with retries. Alerts,
language, disclaimer, sources and schedule blocking are never taken from the LLM."""

import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app.agent.grounding import cited_ids, collect_numbers, find_ungrounded, numbers_in_text
from app.agent.guards import Alert, GuardReport, Thresholds, compute_alerts, no_forecast_alert
from app.agent.i18n import (
    DEGRADED,
    DISCLAIMER,
    NO_PRICE_DATA,
    Lang,
    detect_language,
    localize_alerts,
)
from app.agent.prompts import final_instruction, system_prompt
from app.agent.schemas import (
    AskRequest,
    AskResponse,
    LLMAnswer,
    ScheduleItem,
    SellOrHold,
)
from app.agent.tools import Location, ToolRegistry
from app.config import Settings
from app.errors import SourceUnavailable
from app.llm.base import ChatMessage, LLMClient

log = logging.getLogger("agri.agent")
IST = ZoneInfo("Asia/Kolkata")
_RESTRICTS = {"harvest": "no_harvest", "spray": "no_spraying"}


def parse_answer(raw: str) -> LLMAnswer:
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return LLMAnswer.model_validate(json.loads(text[start : end + 1]))


def check_answer(answer: LLMAnswer, tools: ToolRegistry, known: list[float]) -> list[str]:
    """Problems that must be fixed by the LLM (empty list = acceptable)."""
    errors: list[str] = []
    ids = set(tools.sources)
    cited = cited_ids(answer.answer_text)
    if unknown := sorted(cited - ids):
        errors.append(f"answer_text cites unknown source ids {unknown}")
    if ids and not cited & ids:
        errors.append("answer_text has no citation like [S1]; cite every fact")
    for item in answer.schedule:
        if bad := sorted(set(item.sources) - ids):
            errors.append(f"schedule item cites unknown source ids {bad}")
    if answer.sell_or_hold:
        s = answer.sell_or_hold
        if bad := sorted(set(s.sources) - ids):
            errors.append(f"sell_or_hold cites unknown source ids {bad}")
        if s.recommendation != "insufficient_data" and not s.sources:
            errors.append("sell_or_hold needs source ids unless 'insufficient_data'")
    texts = [answer.answer_text, *(i.description for i in answer.schedule)]
    if answer.sell_or_hold:
        texts.append(answer.sell_or_hold.reasoning)
    bad_nums = sorted({n for t in texts for n in find_ungrounded(t, known)})
    if bad_nums:
        errors.append(
            f"these numbers do not appear in any tool result or alert: {bad_nums}. Remove them or "
            "use the exact values from the tool results"
        )
    return errors


def _apply_restrictions(
    answer: LLMAnswer, report: GuardReport | None, alerts: list[Alert]
) -> list[ScheduleItem]:
    items = [ScheduleItem(**i.model_dump()) for i in answer.schedule]
    if report is None:
        return items
    for item in items:
        need = _RESTRICTS.get(item.activity)
        if need is None or need not in report.restrictions:
            continue
        in_window = item.date is None or (
            report.window_start.date() <= item.date <= report.window_end.date()
        )
        if in_window:
            item.blocked_by = next(a.code for a in alerts if need in a.restrictions)
    return items


async def _gather(
    llm: LLMClient, tools: ToolRegistry, messages: list[ChatMessage], settings: Settings
) -> None:
    for i in range(settings.agent_max_iterations):
        resp = await llm.chat(messages, tools.specs())
        messages.append(
            ChatMessage(role="assistant", content=resp.content, tool_calls=resp.tool_calls)
        )
        if not resp.tool_calls:
            return
        for call in resp.tool_calls:
            result = await tools.execute(call)
            messages.append(
                ChatMessage(
                    role="tool",
                    content=json.dumps(result.content, ensure_ascii=False, default=str),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
        log.info("agent iteration=%d tool_calls=%d", i + 1, len(resp.tool_calls))
    log.warning(
        "agent hit max iterations (%d); forcing final answer", settings.agent_max_iterations
    )


async def run_agent(
    req: AskRequest,
    *,
    llm: LLMClient,
    tools: ToolRegistry,
    thresholds: Thresholds,
    settings: Settings,
    now: datetime,
) -> AskResponse:
    lang: Lang = req.language or detect_language(req.question)
    if req.lat is not None and req.lon is not None:  # user's GPS beats anything the LLM resolves
        tools.location = Location(req.lat, req.lon, f"{req.lat:.2f},{req.lon:.2f}")
    messages = [
        ChatMessage(
            role="system",
            content=system_prompt(lang, now.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")),
        ),
        ChatMessage(role="user", content=req.question),
    ]
    degraded_reason: str | None = None

    # Phase 1: data gathering (LLM decides which tools to call)
    try:
        await _gather(llm, tools, messages, settings)
    except SourceUnavailable as exc:
        degraded_reason = f"LLM unavailable: {exc.user_message()}"
        log.error("agent phase 1 failed: %s", degraded_reason)

    # Deterministic alerts: independent of the LLM, from the forecast for the canonical location
    forecast, why = await tools.ensure_forecast()
    report: GuardReport | None = None
    if forecast is not None:
        report = compute_alerts(forecast, thresholds, now)
        sid = tools.source_for_forecast(forecast)
        raw_alerts = [
            a.model_copy(update={"source_id": sid}) if a.source else a for a in report.alerts
        ]
    else:
        raw_alerts = [no_forecast_alert(why)]
    alerts = localize_alerts(raw_alerts, lang)

    # Phase 2: structured answer, validated with retries
    answer: LLMAnswer | None = None
    if degraded_reason is None:
        known = collect_numbers(tools.facts)
        known += collect_numbers([a.model_dump(mode="json") for a in alerts])
        known += numbers_in_text(req.question)
        known += [v for v in (req.lat, req.lon) if v is not None]
        messages.append(
            ChatMessage(
                role="user",
                content=final_instruction(
                    lang,
                    alerts,
                    list(tools.sources.values()),
                    report.restrictions if report else [],
                ),
            )
        )
        try:
            for attempt in range(settings.agent_json_retries + 1):
                resp = await llm.chat(messages, [], json_mode=True)
                try:
                    candidate = parse_answer(resp.content)
                    problems = check_answer(candidate, tools, known)
                except (ValueError, ValidationError) as exc:
                    candidate, problems = None, [f"not valid per the schema: {str(exc)[:300]}"]
                if candidate is not None and not problems:
                    answer = candidate
                    break
                log.warning("answer rejected (attempt %d): %s", attempt + 1, problems)
                messages.append(ChatMessage(role="assistant", content=resp.content))
                messages.append(
                    ChatMessage(
                        role="user",
                        content="Rejected: "
                        + "; ".join(problems)
                        + ". Return the corrected JSON only.",
                    )
                )
            else:
                degraded_reason = "answer failed validation after retries"
        except SourceUnavailable as exc:
            degraded_reason = f"LLM unavailable: {exc.user_message()}"

    return _build_response(req, lang, answer, alerts, report, tools, llm, degraded_reason)


def _build_response(
    req: AskRequest,
    lang: Lang,
    answer: LLMAnswer | None,
    alerts: list[Alert],
    report: GuardReport | None,
    tools: ToolRegistry,
    llm: LLMClient,
    degraded_reason: str | None,
) -> AskResponse:
    cited: set[str] = {a.source_id for a in alerts if a.source_id}
    if answer is None:
        text, schedule = DEGRADED[lang], []
        sell = SellOrHold(recommendation="insufficient_data", reasoning=NO_PRICE_DATA[lang])
    else:
        text, schedule = answer.answer_text, _apply_restrictions(answer, report, alerts)
        cited |= cited_ids(text) | {s for i in schedule for s in i.sources}
        sell = answer.sell_or_hold or SellOrHold(recommendation="insufficient_data", reasoning="")
        price_cited = any(
            tools.sources[s].kind == "prices" for s in sell.sources if s in tools.sources
        )
        if sell.recommendation != "insufficient_data" and not price_cited:
            # code-level rule: no cited price data => no sell/hold call, whatever the LLM said
            sell = SellOrHold(recommendation="insufficient_data", reasoning=NO_PRICE_DATA[lang])
        cited |= set(sell.sources)
    sources = [s.model_copy(update={"cited": s.id in cited}) for s in tools.sources.values()]
    return AskResponse(
        answer_text=text,
        language=lang,
        alerts=alerts,
        schedule=schedule,
        sell_or_hold=sell,
        sources=sources,
        disclaimer=DISCLAIMER[lang],
        degraded=answer is None,
        degraded_reason=degraded_reason,
        llm=f"{llm.provider}:{llm.model}",
        trace=tools.trace,
    )
