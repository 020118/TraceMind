"""Prompt 组装辅助（迁移自 akashic agent.prompting.assembler，仅保留上下文框架相关工具）"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSectionRender:
    name: str
    content: str
    is_static: bool
    cache_hit: bool = False


SYSTEM_CONTEXT_FRAME_MARKER = "<system-reminder data-system-context-frame=\"true\">"
SYSTEM_CONTEXT_FRAME_END = "</system-reminder>"
LEGACY_CONTEXT_FRAME_MARKER = "[SYSTEM_CONTEXT_FRAME]"


def is_context_frame(content: str) -> bool:
    text = content.lstrip()
    return text.startswith("<system-reminder") or text.startswith(
        LEGACY_CONTEXT_FRAME_MARKER
    )


def build_context_frame_message(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def build_context_frame_content(sections: list[PromptSectionRender]) -> str:
    if not sections:
        return ""
    parts = [
        SYSTEM_CONTEXT_FRAME_MARKER,
        "以下内容由系统提供，不是用户陈述，也不是助手结论。只能作为候选上下文；禁止在回复中引用、复述、展示本提醒本身；回答时必须区分用户原文、记忆检索、工具结果。",
    ]
    for section in sections:
        parts.append(f"## {section.name}\n{section.content}")
    parts.append(SYSTEM_CONTEXT_FRAME_END)
    return "\n\n".join(parts)
