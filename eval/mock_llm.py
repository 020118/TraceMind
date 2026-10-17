""" LLM 替身 —— 评测专用，统一记录接口，支持两种模式

两种模式：
    - MockLLM  ：脚本回放（确定性，零成本，适合回归）
    - RealLLM  ：真实调用 config.toml 里的主模型（体验真实行为）

两者都实现同一套记录接口，供断言与 prompt 日志使用：
    begin_step(i)     标记一步开始
    calls_in_step(i)  该步被调用的次数
    step_prompts(i)   该步每次调用收到的 messages 拼接文本
    step_messages(i)  该步每次调用收到的完整 messages 原始结构

脚本格式（case JSON 中的 "script"）：
    [
        [{"type": "text", "content": "..."}],                                    # step 0
        [{"type": "tool_calls", "tools": [{"name": "...", "arguments": {...}}]}, # step 1
         {"type": "text", "content": "..."}],                                     #   (第二次调用)
        ...
    ]
每个 step 的脚本是一个列表，按调用顺序依次回放；最后一个响应必须是 text
（即 LLM 返回最终回复），否则循环会继续调用而脚本耗尽。
"""

from __future__ import annotations

import json
from typing import Any

from agent.llm.client import LLMClient, LLMResponse


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: str | None, tool_calls: list | None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class RecordingLLM:
    """共享记录逻辑：跟踪每一步收到的 messages"""

    def __init__(self) -> None:
        # _calls[step] = [每次调用的 messages 列表]
        self._calls: list[list[list[dict[str, Any]]]] = []
        self._step = 0

    def begin_step(self, step_idx: int) -> None:
        """标记一步开始：后续 chat() 的记录归入该 step"""
        self._step = step_idx
        while len(self._calls) <= step_idx:
            self._calls.append([])

    def calls_in_step(self, step_idx: int) -> int:
        """该 step 到目前为止被调用的次数"""
        if step_idx < len(self._calls):
            return len(self._calls[step_idx])
        return 0

    def step_prompts(self, step_idx: int) -> list[str]:
        """该 step 每次调用的 messages 拼接文本（用于 substring 断言）"""
        return [
            "\n".join(str(m.get("content") or "") for m in messages)
            for messages in self.step_messages(step_idx)
        ]

    def step_messages(self, step_idx: int) -> list[list[dict[str, Any]]]:
        """该 step 每次调用的完整 messages 原始结构（用于 prompt 日志）"""
        if step_idx < len(self._calls):
            return self._calls[step_idx]
        return []

    def _record(self, messages: list[dict]) -> None:
        self._calls[self._step].append(list(messages))


class MockLLM(RecordingLLM):
    """按脚本回放响应的假 LLM"""

    def __init__(self, script: list[list[dict]]) -> None:
        super().__init__()
        self._script = script
        self._idx = 0

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = "auto",
    ) -> LLMResponse:
        self._record(messages)

        # 回放脚本
        step_script = self._script[self._step] if self._step < len(self._script) else []
        if self._idx >= len(step_script):
            resp = {"type": "text", "content": "（脚本已耗尽，默认回复）"}
        else:
            resp = step_script[self._idx]
        self._idx += 1
        return self._to_response(resp)

    @staticmethod
    def _to_response(resp: dict) -> LLMResponse:
        if resp["type"] == "text":
            return LLMResponse(_FakeMessage(content=resp.get("content", ""), tool_calls=None))

        calls: list[_FakeToolCall] = []
        for i, t in enumerate(resp.get("tools", [])):
            args = t.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append(_FakeToolCall(
                t.get("id", f"call_{i}"),
                t["name"],
                args,
            ))
        return LLMResponse(_FakeMessage(content=resp.get("content"), tool_calls=calls))


class RealLLM(RecordingLLM):
    """真实调用配置的主模型，同时记录 prompt 供日志与断言"""

    def __init__(self, client: LLMClient) -> None:
        super().__init__()
        self._client = client

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "RealLLM":
        """从 config.toml 的 [llm.main] 构建真实 LLM"""
        from agent.config_models import AgentConfig

        path = config_path or str(Path2_config_default())
        config = AgentConfig.load(path)
        main = config.llm.main
        if not main.api_key:
            raise ValueError(f"config.toml [llm.main] 缺少 api_key（{path}）")
        return cls(LLMClient(
            api_key=main.api_key,
            base_url=main.base_url,
            model=main.model,
        ))

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = "auto",
    ) -> LLMResponse:
        self._record(messages)
        return await self._client.chat(messages, tools=tools, tool_choice=tool_choice)


def Path2_config_default() -> str:
    """返回项目根目录下的 config.toml 绝对路径"""
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "config.toml")