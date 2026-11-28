"""独立只读 AI Code Reviewer 工具。

核心设计：
- 不继承当前 Coder 对话历史，另起干净的独立 LLM 上下文，消除自我辩护/锚定偏见。
- 只读取当前工作区的 git status + git diff（含脱敏）与测试结果摘要。
- 要求模型输出结构化 JSON（approved / risk_level / summary / issues / suggestions）。
- 只有 approved=True 时才允许进入后续 git_commit_and_push 人工审批交付。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent.security import redact_secrets
from agent.tools.base import BaseTool, ToolRegistry
from agent.tools.git_tools import _run_git

logger = logging.getLogger(__name__)

_MAX_REVIEW_DIFF_CHARS = 24_000

_REVIEWER_SYSTEM_PROMPT = """你是一位严谨的高级代码评审专家（Senior Code Reviewer）。
你处于独立只读评审上下文中，不参与编写代码，唯一职责是审查即将交付的 git diff。

请重点检查：
1. 逻辑错误、边界条件遗漏、异常处理缺失、并发/状态竞态问题；
2. 安全隐患（SSRF、路径穿越、注入、硬编码密钥、越权、破坏现有保护边界）；
3. 是否破坏既有接口兼容性，以及测试覆盖是否充分。

必须只返回一个合法 JSON 对象，不要包含额外解释，格式严格如下：
{
  "approved": true 或 false,
  "risk_level": "low" | "medium" | "high",
  "summary": "1-2 句总体评审结论",
  "issues": [
    {"severity": "high|medium|low", "file": "文件路径", "detail": "具体问题描述与修复建议"}
  ],
  "suggestions": ["可选的非阻塞改进建议"]
}

判定标准：只要存在任意 severity 为 "high" 的问题，或者存在明显功能 Bug/安全漏洞，approved 必须为 false。"""


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class CodeReviewDiffTool(BaseTool):
    name = "code_review_diff"
    description = (
        "使用独立只读的 Reviewer 上下文审查当前工作区的 git diff，输出结构化评审结论（approved/issues）。"
        "在提交推送代码前必须调用此工具；若未通过，请先修复高危问题后再重新评审。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "本次代码改动要实现的目标，供 Reviewer 对照检查",
            },
            "test_summary": {
                "type": "string",
                "description": "已运行的语法检查/单测/构建结果摘要",
            },
        },
        "required": ["goal"],
    }

    def __init__(
        self,
        workspace: Path,
        provider: Any,
        *,
        model: str = "",
    ) -> None:
        self._workspace = workspace.expanduser().resolve()
        self._provider = provider
        self._model = model

    async def execute(
        self,
        goal: str,
        test_summary: str = "",
        **_: Any,
    ) -> str:
        rc_s, status_out, err_s = await _run_git(self._workspace, ["status", "--short"])
        if rc_s != 0:
            return f"Code Review 失败：无法获取 Git 状态：{err_s.strip()}"

        # 合并未暂存与已暂存 diff，确保不遗漏
        _, unstaged, _ = await _run_git(self._workspace, ["diff"])
        _, staged, _ = await _run_git(self._workspace, ["diff", "--cached"])
        combined_diff = "\n".join(part for part in (staged.strip(), unstaged.strip()) if part).strip()

        if not combined_diff and not status_out.strip():
            return "Code Review 未执行：当前工作区没有任何代码改动。"

        safe_diff = redact_secrets(combined_diff)
        if len(safe_diff) > _MAX_REVIEW_DIFF_CHARS:
            safe_diff = (
                safe_diff[:_MAX_REVIEW_DIFF_CHARS]
                + f"\n\n... [diff 已截断，原始长度 {len(combined_diff)} 字符]"
            )

        user_prompt = (
            f"## 变更目标\n{goal}\n\n"
            f"## 自动化测试/构建结果\n{test_summary.strip() or '（调用方未提供测试摘要，请重点提醒补充测试）'}\n\n"
            f"## Git Status\n{status_out.strip() or '无'}\n\n"
            f"## Git Diff\n```diff\n{safe_diff or '（仅有新增未跟踪文件，尚未 git add）'}\n```"
        )

        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
                model=self._model or None,
                max_tokens=1500,
                tool_choice="none",
            )
        except Exception as exc:
            logger.exception("Independent code review LLM call failed")
            return f"Code Review 失败：独立评审模型调用异常：{exc}"

        raw_reply = (getattr(resp, "content", None) or "").strip()
        parsed = _extract_json_object(raw_reply)
        if parsed is None:
            return (
                "Code Review 结果：未通过（评审模型未返回合法结构化 JSON，请重试或人工检查）。\n"
                f"原始输出：{raw_reply[:600]}"
            )

        approved = bool(parsed.get("approved", False))
        risk_level = str(parsed.get("risk_level", "medium")).lower()
        summary = str(parsed.get("summary", "")).strip() or "无摘要"
        issues = parsed.get("issues") or []
        suggestions = parsed.get("suggestions") or []

        # 防御性兜底：只要存在 high 级别问题，强制不通过
        if any(
            isinstance(item, dict) and str(item.get("severity", "")).lower() == "high"
            for item in issues
        ):
            approved = False

        status_text = "通过 (approved=true)" if approved else "未通过 (approved=false)"
        lines = [
            f"独立 AI Code Review 结论：{status_text}",
            f"风险等级：{risk_level}",
            f"评审摘要：{summary}",
        ]
        if issues:
            lines.append("发现的问题：")
            for idx, item in enumerate(issues, 1):
                if isinstance(item, dict):
                    sev = str(item.get("severity", "medium")).upper()
                    file_path = str(item.get("file", "未知文件"))
                    detail = str(item.get("detail", ""))
                    lines.append(f"  {idx}. [{sev}] {file_path}: {detail}")
                else:
                    lines.append(f"  {idx}. {item}")
        if suggestions:
            lines.append("改进建议：")
            for item in suggestions:
                lines.append(f"  - {item}")
        if not approved:
            lines.append("下一步：请根据上述问题修复代码并重新运行测试，然后再次调用 code_review_diff。")
        else:
            lines.append("下一步：可以调用 git_commit_and_push 发起人工确认并推送到功能分支。")
        return "\n".join(lines)


def register_code_review_tool(
    registry: ToolRegistry,
    workspace: Path,
    provider: Any,
    *,
    model: str = "",
) -> None:
    registry.register(
        CodeReviewDiffTool(workspace, provider, model=model),
        risk="read-only",
        aliases=("代码评审", "code review", "审查代码", "review diff"),
    )


__all__ = ["CodeReviewDiffTool", "register_code_review_tool"]
