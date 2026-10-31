"""Ollama /api/chat client (local models). Temperature 0 for repeatable behaviour."""

import json
from collections.abc import Sequence
from typing import Any

import httpx

from app.errors import MalformedPayload
from app.llm.base import ChatMessage, LLMResponse, ToolCall, ToolSpec, post_json


def _to_wire(m: ChatMessage) -> dict[str, Any]:
    if m.role == "assistant" and m.tool_calls:
        calls = [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
        return {"role": "assistant", "content": m.content, "tool_calls": calls}
    if m.role == "tool":
        return {"role": "tool", "content": m.content, "tool_name": m.name or ""}
    return {"role": m.role, "content": m.content}


class OllamaClient:
    provider = "ollama"

    def __init__(
        self, client: httpx.AsyncClient, base_url: str, model: str, timeout: float
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self._timeout = timeout

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_wire(m) for m in messages],
            "stream": False,
            "options": {"temperature": 0},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        if json_mode:
            payload["format"] = "json"
        data = await post_json(
            self._client, self._url, source="LLM (ollama)", payload=payload, timeout=self._timeout
        )
        msg = data.get("message")
        if not isinstance(msg, dict):
            raise MalformedPayload("LLM (ollama)", "missing 'message'")
        calls: list[ToolCall] = []
        for i, raw in enumerate(msg.get("tool_calls") or []):
            fn = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise MalformedPayload("LLM (ollama)", "bad tool_call entry")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_invalid_json": args[:200]}
            calls.append(ToolCall(id=f"call_{i}", name=fn["name"], arguments=args or {}))
        return LLMResponse(content=msg.get("content") or "", tool_calls=calls)
