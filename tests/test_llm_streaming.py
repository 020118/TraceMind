from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.llm.client import LLMClient


def _chunk(*, content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls or [],
                )
            )
        ]
    )


@pytest.mark.asyncio
async def test_llm_client_streams_text_and_reassembles_tool_call() -> None:
    chunks = [
        _chunk(content="你"),
        _chunk(content="好"),
        _chunk(
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id="call-1",
                    function=SimpleNamespace(name="read_", arguments='{"pa'),
                )
            ]
        ),
        _chunk(
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id=None,
                    function=SimpleNamespace(name="file", arguments='th":"README.md"}'),
                )
            ]
        ),
    ]

    class _Completions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True

            async def generate():
                for chunk in chunks:
                    yield chunk

            return generate()

    client = LLMClient("test", "http://example.invalid", "test-model")
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )
    deltas: list[str] = []

    async def on_delta(delta: str) -> None:
        deltas.append(delta)

    response = await client.chat(
        [{"role": "user", "content": "hello"}],
        on_content_delta=on_delta,
    )

    assert deltas == ["你", "好"]
    assert response.content == "你好"
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].function.name == "read_file"
    assert response.tool_calls[0].function.arguments == '{"path":"README.md"}'
