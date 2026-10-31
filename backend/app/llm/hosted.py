"""OpenAI-compatible /chat/completions client (any hosted provider). Temperature 0."""

import json
from collections.abc import Sequence
from typing import Any

import httpx

from app.errors import MalformedPayload
from app.llm.base import ChatMessage, LLMResponse, ToolCall, ToolSpec, post_json

SOURCE = "LLM (hosted)"


def _to_wire(m: ChatMessage) -> dict[str, Any]:
    if m.role == "assistant" and m.tool_calls:
        calls = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in m.tool_calls
        ]
        return {"role": "assistant", "content": m.content or None, "tool_calls": calls}
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content}
    return {"role": m.role, "content": m.content}


class HostedClient:
    provider = "hosted"

    def __init__(
        self, client: httpx.AsyncClient, base_url: str, api_key: str, model: str, timeout: float
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
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
            "temperature": 0,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = await post_json(
            self._client,
            self._url,
            source=SOURCE,
            payload=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise MalformedPayload(SOURCE, "missing choices[0].message") from exc
        calls: list[ToolCall] = []
        for i, raw in enumerate(msg.get("tool_calls") or []):
            try:
                fn = raw["function"]
                name = fn["name"]
            except (KeyError, TypeError) as exc:
                raise MalformedPayload(SOURCE, "bad tool_call entry") from exc
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = {"_invalid_json": str(fn.get("arguments"))[:200]}
            calls.append(ToolCall(id=str(raw.get("id") or f"call_{i}"), name=name, arguments=args))
        return LLMResponse(content=msg.get("content") or "", tool_calls=calls)
