"""Request/response models. LLMAnswer is all the LLM may produce; everything else in AskResponse
(alerts, language, disclaimer, sources, blocked flags) is filled in by code."""

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.guards import Alert, AlertCode
from app.agent.i18n import Lang

Activity = Literal["harvest", "spray", "irrigate", "sow", "fertilize", "sell", "other"]
Recommendation = Literal["sell_now", "hold", "insufficient_data"]
SourceKind = Literal["weather", "prices", "soil", "geocode", "knowledge"]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    language: Lang | None = None  # default: detected from the script of `question`
    lat: float | None = Field(default=None, ge=-90, le=90)  # optional GPS; makes alerts exact
    lon: float | None = Field(default=None, ge=-180, le=180)


class LLMScheduleItem(BaseModel):
    date: dt.date | None = None
    activity: Activity
    description: str = Field(min_length=1, max_length=400)
    sources: list[str] = Field(default_factory=list)


class ScheduleItem(LLMScheduleItem):
    blocked_by: AlertCode | None = None  # set by code when a safety alert forbids this activity


class SellOrHold(BaseModel):
    recommendation: Recommendation
    reasoning: str = Field(max_length=800)
    sources: list[str] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    answer_text: str = Field(min_length=1, max_length=4000)
    schedule: list[LLMScheduleItem] = Field(default_factory=list, max_length=10)
    sell_or_hold: SellOrHold | None = None


class SourceRef(BaseModel):
    id: str
    kind: SourceKind
    name: str
    url: str
    fetched_at: dt.datetime | None = None
    cache_status: Literal["live", "cached", "stale"] | None = None
    detail: str | None = None  # e.g. "p. 12", "ingested 2026-10-31"
    cited: bool = False


class ToolTrace(BaseModel):
    name: str
    arguments: dict[str, object]
    ok: bool
    latency_ms: int
    source_ids: list[str]


class AskResponse(BaseModel):
    answer_text: str
    language: Lang
    alerts: list[Alert]
    schedule: list[ScheduleItem]
    sell_or_hold: SellOrHold
    sources: list[SourceRef]
    disclaimer: str
    degraded: bool = False  # True: no verified LLM answer; alerts + sources only
    degraded_reason: str | None = None
    llm: str  # "<provider>:<model>" (for audit; alerts do not depend on it)
    trace: list[ToolTrace]
