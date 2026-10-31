"""Ollama / hosted clients against a mock transport: request shape, parsing, error mapping."""

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.errors import MalformedPayload, NotConfigured, SourceUnavailable
from app.llm.base import ChatMessage, ToolCall, ToolSpec, build_llm
from app.llm.hosted import HostedClient
from app.llm.ollama import OllamaClient
from tests.helpers import make_settings

TOOL = ToolSpec(
    name="get_weather", description="d", parameters={"type": "object", "properties": {}}
)
HISTORY = [
    ChatMessage(role="system", content="sys"),
    ChatMessage(role="user", content="q"),
    ChatMessage(
        role="assistant", tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"lat": 1.0})]
    ),
    ChatMessage(role="tool", content='{"x": 1}', tool_call_id="c1", name="get_weather"),
]


def chat(client_cls: Any, handler: Any, **kw: Any) -> Any:
    async def go() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            c: Any
            if client_cls is OllamaClient:
                c = OllamaClient(http, "http://ollama.test/", "m1", 5)
            else:
                c = HostedClient(http, "https://api.test/v1/", "KEY", "m2", 5)
            return await c.chat(
                HISTORY, kw.get("tools", [TOOL]), json_mode=kw.get("json_mode", False)
            )

    return asyncio.run(go())


def test_ollama_request_and_tool_call_parsing() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(url=str(req.url), body=json.loads(req.content))
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"lat": 12.5, "lon": 76.9}}}]}  # fmt: skip
        return httpx.Response(200, json={"message": msg})

    r = chat(OllamaClient, handler, json_mode=True)
    b = seen["body"]
    assert seen["url"] == "http://ollama.test/api/chat" and b["stream"] is False
    assert b["options"] == {"temperature": 0} and b["format"] == "json" and b["model"] == "m1"
    assert b["tools"][0]["function"]["name"] == "get_weather"
    assert b["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"lat": 1.0}
    assert b["messages"][3] == {"role": "tool", "content": '{"x": 1}', "tool_name": "get_weather"}
    assert r.tool_calls[0].name == "get_weather" and r.tool_calls[0].arguments == {
        "lat": 12.5,
        "lon": 76.9,
    }


def test_ollama_plain_text_and_no_tools_key() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hello"}})

    r = chat(OllamaClient, handler, tools=[])
    assert r.content == "hello" and r.tool_calls == [] and "tools" not in seen["body"]
    assert "format" not in seen["body"]


def test_hosted_request_and_string_arguments_parsing() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(
            url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content)
        )
        msg = {"content": None, "tool_calls": [{"id": "abc", "type": "function",
               "function": {"name": "get_weather", "arguments": '{"lat": 12.5}'}}]}  # fmt: skip
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    r = chat(HostedClient, handler, json_mode=True)
    b = seen["body"]
    assert seen["url"] == "https://api.test/v1/chat/completions" and seen["auth"] == "Bearer KEY"
    assert b["temperature"] == 0 and b["response_format"] == {"type": "json_object"}
    assert b["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"lat": 1.0}'
    assert b["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": '{"x": 1}'}
    assert (r.tool_calls[0].id, r.tool_calls[0].arguments) == (
        "abc",
        {"lat": 12.5},
    ) and r.content == ""


def test_invalid_tool_argument_json_is_surfaced_not_crashed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        msg = {
            "content": "",
            "tool_calls": [{"id": "1", "function": {"name": "t", "arguments": "{oops"}}],
        }
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    assert chat(HostedClient, handler).tool_calls[0].arguments == {"_invalid_json": "{oops"}


@pytest.mark.parametrize("cls", [OllamaClient, HostedClient])
def test_http_and_payload_errors_map_to_source_unavailable(cls: Any) -> None:
    with pytest.raises(SourceUnavailable) as e:
        chat(cls, lambda _: httpx.Response(500, text="secret internal detail"))
    assert e.value.kind == "http_error" and "secret" not in e.value.user_message()
    with pytest.raises(MalformedPayload):
        chat(cls, lambda _: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(MalformedPayload):
        chat(cls, lambda _: httpx.Response(200, text="<html>"))

    def boom(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(SourceUnavailable) as e2:
        chat(cls, boom)
    assert e2.value.kind == "network"


def test_provider_selection_and_config_errors() -> None:
    http = httpx.AsyncClient()
    assert isinstance(
        build_llm(make_settings(llm_provider="ollama", ollama_model="m"), http), OllamaClient
    )
    hosted = make_settings(
        llm_provider="hosted",
        hosted_llm_base_url="https://x/v1",
        hosted_llm_api_key="k",
        hosted_llm_model="m",
    )
    assert isinstance(build_llm(hosted, http), HostedClient)
    with pytest.raises(NotConfigured, match="OLLAMA_MODEL"):
        build_llm(make_settings(llm_provider="ollama"), http)
    with pytest.raises(NotConfigured, match="HOSTED_LLM_API_KEY"):
        build_llm(
            make_settings(
                llm_provider="hosted", hosted_llm_base_url="https://x/v1", hosted_llm_model="m"
            ),
            http,
        )
