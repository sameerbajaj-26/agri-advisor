"""Provider-neutral chat/tool-calling types. Ollama and hosted clients translate to/from these,
so the agent loop (and the safety alerts) never depend on which model is plugged in."""

from collections.abc import Sequence
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import MalformedPayload, NotConfigured, SourceUnavailable


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant -> requested calls
    tool_call_id: str | None = None  # tool -> which call this answers
    name: str | None = None  # tool -> tool name


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMClient(Protocol):
    provider: str
    model: str

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        *,
        json_mode: bool = False,
    ) -> LLMResponse: ...


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    source: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float,
) -> dict[str, Any]:
    """POST and map every failure to SourceUnavailable (never leaks the URL or API key)."""
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise SourceUnavailable(source, "timeout", f"no response within {timeout:.0f}s") from exc
    except httpx.HTTPError as exc:
        raise SourceUnavailable(source, "network", "connection failed") from exc
    if resp.status_code >= 400:
        raise SourceUnavailable(
            source, "http_error", f"HTTP {resp.status_code}", status_code=resp.status_code
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise MalformedPayload(source, "response was not JSON") from exc
    if not isinstance(data, dict):
        raise MalformedPayload(source, "expected a JSON object")
    return data


def build_llm(settings: Settings, client: httpx.AsyncClient) -> LLMClient:
    """Select the model backend from LLM_PROVIDER."""
    from app.llm.hosted import HostedClient
    from app.llm.ollama import OllamaClient

    if settings.llm_provider == "ollama":
        if not settings.ollama_model:
            raise NotConfigured("LLM (ollama)", "OLLAMA_MODEL is not set (pull one first)")
        return OllamaClient(
            client, settings.ollama_url, settings.ollama_model, settings.llm_timeout_s
        )
    missing = [
        n
        for n, v in (
            ("HOSTED_LLM_BASE_URL", settings.hosted_llm_base_url),
            ("HOSTED_LLM_API_KEY", settings.hosted_llm_api_key),
            ("HOSTED_LLM_MODEL", settings.hosted_llm_model),
        )
        if not v
    ]
    if missing:
        raise NotConfigured("LLM (hosted)", f"{', '.join(missing)} not set")
    assert (
        settings.hosted_llm_base_url and settings.hosted_llm_api_key and settings.hosted_llm_model
    )
    return HostedClient(
        client,
        settings.hosted_llm_base_url,
        settings.hosted_llm_api_key.get_secret_value(),
        settings.hosted_llm_model,
        settings.llm_timeout_s,
    )
