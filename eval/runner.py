""" Agent 边界评测运行器 —— 跑真实 agent_loop，量化边界指标

用法：
    python -m eval.runner                    # 跑全部用例（默认 mock，零成本）
    python -m eval.runner --real             # 用真实 LLM（config.toml [llm.main]）跑全部
    python -m eval.runner --real --case 会话  # 真实 LLM 只跑指定用例
    python -m eval.runner --list             # 只列出用例

设计：
    - 评测集在 eval/cases/*.json，每个 case 描述：步骤（发给 agent 的消息序列）、
      LLM 脚本（按步骤回放）、断言（对 LLM 收到的 prompt / 工具调用 / 回复的检查）
    - 运行器跑真实的 agent_loop（消息总线 + 工具 + 记忆全走真链路）
    - LLM 有两种模式：MockLLM 脚本回放（默认，确定性回归）/ RealLLM 真实调用（体验真实行为）
    - 每一步发给 LLM 的完整 prompt 会写入 eval/logs/，供人工核对"agent 到底看到了什么"
    - 输出每个 case 的 pass/fail，并汇总成报告，失败项即"边界/痛点"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bus.event_bus import EventBus
from agent.bus.events import InboundMessage, OutboundMessage
from agent.bus.queue import MessageBus
from agent.core.agent_loop import agent_loop
from agent.memory.service import MemoryService
from agent.tools.base import BaseTool, ToolRegistry

from eval.mock_llm import MockLLM, RealLLM, RecordingLLM

logger = logging.getLogger(__name__)

CASES_DIR = Path(__file__).resolve().parent / "cases"
LOGS_DIR = Path(__file__).resolve().parent / "logs"


# ── 评测专用工具 ───────────────────────────────────────────


class SlowProbeTool(BaseTool):
    """慢工具：并发探针。记录最大并发数，用于测多工具是否并行执行。"""

    name = "slow_probe"
    description = "慢速探针，用于测试工具并行执行"
    parameters = {
        "type": "object",
        "properties": {"tag": {"type": "string", "description": "探针标签"}},
        "required": [],
    }

    def __init__(self, recorder: dict, delay: float = 0.3) -> None:
        self._recorder = recorder
        self._delay = delay
        self._active = 0

    async def execute(self, tag: str = "") -> str:
        self._active += 1
        self._recorder["max_concurrency"] = max(
            self._recorder.get("max_concurrency", 0), self._active
        )
        await asyncio.sleep(self._delay)
        self._active -= 1
        self._recorder["tool_calls"].append({"name": self.name, "args": {"tag": tag}})
        return f"probe[{tag}] done in {self._delay}s"


class EchoTool(BaseTool):
    """回声工具：返回传入内容。"""

    name = "echo_tool"
    description = "回声工具，返回传入的文本"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    async def execute(self, text: str = "") -> str:
        self._recorder["tool_calls"].append({"name": self.name, "args": {"text": text}})
        return f"echo:{text}"


class FailTool(BaseTool):
    """失败工具：抛出异常，用于测 agent 对工具异常的处理。"""

    name = "fail_tool"
    description = "总是执行失败的工具，用于测试异常路径"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    async def execute(self) -> str:
        self._recorder["tool_calls"].append({"name": self.name, "args": {}})
        raise RuntimeError("模拟的工具内部错误")


def _build_registry(recorder: dict) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(SlowProbeTool(recorder))
    reg.register(EchoTool(recorder))
    reg.register(FailTool(recorder))
    return reg


class _FakeMemory(MemoryService):
    """空记忆服务：模拟未启用记忆的场景（快进到 build_memory_context 返回空）。"""

    def __init__(self) -> None:
        pass

    async def build_memory_context(self, **kwargs) -> str:
        return ""

    async def extract_facts(self, **kwargs) -> list:
        return []

    async def add_memory(self, **kwargs) -> int:
        return -1


# ── 断言检查 ────────────────────────────────────────────────


def _check_assertion(
    assertion: dict,
    llm: MockLLM,
    outbound_msgs: list[OutboundMessage],
    recorder: dict,
    memory_service: MemoryService | None,
    tools: ToolRegistry,
) -> tuple[bool, str]:
    """执行单条断言，返回 (是否通过, 说明)"""
    kind = assertion["type"]
    step = assertion.get("step", 0)

    if kind == "llm_prompt_contains":
        # 某 step 的 LLM prompt（历史 + 当前消息）应包含指定子串
        prompts = llm.step_prompts(step)
        blob = "\n".join(prompts)
        for sub in assertion["substrings"]:
            if sub not in blob:
                return False, f"LLM prompt 缺少子串 {sub!r}"
        return True, "LLM prompt 包含全部子串"

    if kind == "llm_prompt_not_contains":
        prompts = llm.step_prompts(step)
        blob = "\n".join(prompts)
        for sub in assertion["substrings"]:
            if sub in blob:
                return False, f"LLM prompt 意外包含子串 {sub!r}"
        return True, "LLM prompt 未包含禁用子串"

    if kind == "tool_called":
        # 指定工具被调用，且参数匹配
        for call in recorder.get("tool_calls", []):
            if call["name"] != assertion["name"]:
                continue
            args = assertion.get("args", {})
            if all(call["args"].get(k) == v for k, v in args.items()):
                return True, f"工具 {assertion['name']} 被正确调用"
        return False, f"工具 {assertion['name']} 未被正确调用"

    if kind == "tool_not_called":
        for call in recorder.get("tool_calls", []):
            if call["name"] == assertion["name"]:
                return False, f"工具 {assertion['name']} 不应被调用但被调用了"
        return True, f"工具 {assertion['name']} 未被调用"

    if kind == "max_concurrency":
        # 并发探针的最大并发数应达到目标（测多工具并行）
        got = recorder.get("max_concurrency", 0)
        target = assertion["target"]
        if got < target:
            return False, f"最大并发 {got} < 目标 {target}（工具是串行执行的）"
        return True, f"最大并发 {got} >= {target}"

    if kind == "outbound_contains":
        if step >= len(outbound_msgs):
            return False, f"第 {step} 轮没有产出出站消息"
        content = outbound_msgs[step].content
        for sub in assertion["substrings"]:
            if sub not in content:
                return False, f"出站回复缺少子串 {sub!r}"
        return True, "出站回复包含全部子串"

    if kind == "outbound_not_contains":
        if step >= len(outbound_msgs):
            return False, f"第 {step} 轮没有产出出站消息"
        content = outbound_msgs[step].content
        for sub in assertion["substrings"]:
            if sub in content:
                return False, f"出站回复意外包含子串 {sub!r}"
        return True, "出站回复未包含禁用子串"

    if kind == "llm_calls":
        # 某 step 的 LLM 调用次数（测工具循环是否收敛/死循环）
        got = llm.calls_in_step(step)
        target = assertion["max"]
        if got > target:
            return False, f"LLM 调用 {got} 次 > 上限 {target}（疑似死循环）"
        return True, f"LLM 调用 {got} 次 <= {target}"

    if kind == "no_memory_extracted":
        # 记忆萃取结果应满足条件（需要记忆服务代理）
        raise NotImplementedError("no_memory_extracted 尚未实现")

    raise NotImplementedError(f"未知断言类型: {kind}")


# ── 单 case 执行 ────────────────────────────────────────────


async def run_single_case(case: dict, *, use_real_llm: bool = False) -> dict:
    """跑单个 case，返回结果 dict

    use_real_llm=True 时用真实 LLM（忽略 script，按真实行为推理）；
    否则用 MockLLM 脚本回放。
    """
    steps: list[dict] = case["steps"]
    script: list[list[dict]] = case.get("script", [])
    assertions: list[dict] = case.get("assertions", [])

    bus = MessageBus()
    events = EventBus()
    recorder: dict = {"tool_calls": [], "max_concurrency": 0}
    tools = _build_registry(recorder)
    memory_service: MemoryService | None = (
        _FakeMemory() if case.get("memory_enabled", False) else None
    )

    if use_real_llm:
        llm: RecordingLLM = RealLLM.from_config()
    else:
        llm = MockLLM(script)

    # 收集出站消息
    outbound_msgs: list[OutboundMessage] = []

    async def _collect_outbound() -> None:
        while True:
            outbound_msgs.append(await bus.consume_outbound())

    agent_task = asyncio.create_task(
        agent_loop(bus, llm, events, tools=tools, memory_service=memory_service)
    )
    collector_task = asyncio.create_task(_collect_outbound())

    try:
        for i, step in enumerate(steps):
            llm.begin_step(i)
            msg = InboundMessage(
                channel=step["channel"],
                chat_id=step.get("chat_id", "chat"),
                sender_id=step.get("sender_id", "user"),
                content=step["content"],
            )
            await bus.publish_inbound(msg)

            # 等待本轮处理完：真实 LLM 按出站消息判定；mock 按脚本调用次数判定
            if use_real_llm:
                timeout = 60
                done = lambda i=i: len(outbound_msgs) > i or llm.calls_in_step(i) > 0
            else:
                timeout = 15
                expected_calls = len(script[i]) if i < len(script) else 0
                done = lambda i=i: (
                    llm.calls_in_step(i) >= expected_calls
                    or len(outbound_msgs) > i
                )
            await _wait_for(done, timeout=timeout, label=f"step{i} 处理完成")
            await asyncio.sleep(0.1)  # 留一点时间让出站消息落袋
    finally:
        agent_task.cancel()
        collector_task.cancel()
        await asyncio.gather(agent_task, collector_task, return_exceptions=True)
        await events.close()

    # 写 prompt 日志（供人工核对 agent 每步看到了什么）
    log_path = _write_prompt_log(case, llm)

    # 执行断言
    results = []
    for assertion in assertions:
        try:
            passed, detail = _check_assertion(
                assertion, llm, outbound_msgs, recorder, memory_service, tools
            )
        except NotImplementedError as e:
            passed, detail = False, str(e)
        results.append({
            "desc": assertion.get("desc", assertion.get("type", "")),
            "passed": passed,
            "detail": detail,
        })

    return {
        "name": case.get("name", "未命名"),
        "category": case.get("category", "未分类"),
        "description": case.get("description", ""),
        "mode": "real" if use_real_llm else "mock",
        "assertions": results,
        "outbound_count": len(outbound_msgs),
        "llm_calls_total": sum(llm.calls_in_step(i) for i in range(len(steps))),
        "log_path": str(log_path),
    }


def _write_prompt_log(case: dict, llm: RecordingLLM) -> Path:
    """把每个 step 发给 LLM 的完整 messages 写入 eval/logs/<name>.md"""
    steps: list[dict] = case["steps"]
    lines = [
        f"# {case.get('name', '未命名')} — LLM Prompt 日志",
        "",
        f"说明: 每个 step 记录 agent 实际发给 LLM 的完整 messages（含 system/history/当前消息）。",
        "",
    ]
    for i, step in enumerate(steps):
        lines.append(f"## Step {i}: [{step['channel']} / {step.get('chat_id','')}] {step['content']}")
        lines.append("")
        msgs_list = llm.step_messages(i)
        if not msgs_list:
            lines.append("（该 step 未调用 LLM）")
            lines.append("")
            continue
        for c, messages in enumerate(msgs_list):
            lines.append(f"### LLM 调用 #{c + 1}")
            lines.append("")
            for m in messages:
                role = m.get("role", "?")
                content = m.get("content") or ""
                lines.append(f"- **{role}**: {content}")
                lines.append("")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{case['_path'].removesuffix('.json')}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def _wait_for(pred, timeout: float, label: str) -> None:
    """轮询等待条件满足或超时"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if pred():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"等待超时: {label}")


# ── 加载与报告 ──────────────────────────────────────────────


def load_cases() -> list[dict]:
    """从 eval/cases/*.json 加载全部用例"""
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
        case["_path"] = path.name
        cases.append(case)
    return cases


def render_report(results: list[dict]) -> str:
    """生成可读报告文本"""
    lines = []
    total = sum(len(r["assertions"]) for r in results)
    passed = sum(sum(1 for a in r["assertions"] if a["passed"]) for r in results)
    failed = total - passed

    lines.append("=" * 64)
    lines.append("Agent 边界评测报告")
    lines.append("=" * 64)
    lines.append(f"用例: {len(results)} 个 | 断言: {total} 条 | 通过: {passed} | 失败: {failed}")
    lines.append("")

    for r in results:
        status = "PASS" if all(a["passed"] for a in r["assertions"]) else "FAIL"
        mode_tag = "真实LLM" if r["mode"] == "real" else "mock"
        lines.append(f"[{status}] {r['name']}  ({r['category']} · {mode_tag})")
        lines.append(f"    {r['description']}")
        if r["assertions"]:
            for a in r["assertions"]:
                mark = "✓" if a["passed"] else "✗"
                lines.append(f"    {mark} {a['desc']}")
                if not a["passed"]:
                    lines.append(f"        → {a['detail']}")
        lines.append(f"    信息: 出站消息 {r['outbound_count']} 条 | LLM 调用 {r['llm_calls_total']} 次")
        lines.append(f"    prompt 日志: {r['log_path']}")
        lines.append("")

    lines.append("-" * 64)
    lines.append("失败项 = 当前边界/痛点，可作为改进方向与面试素材")
    lines.append("=" * 64)
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 边界评测")
    parser.add_argument("--case", help="只跑匹配该名称的用例")
    parser.add_argument("--list", action="store_true", help="只列出用例")
    parser.add_argument("--real", action="store_true", help="用真实 LLM 跑（读取 config.toml [llm.main]）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("eval").setLevel(logging.WARNING)

    cases = load_cases()
    if not cases:
        print(f"未找到评测用例（{CASES_DIR}）")
        return

    if args.list:
        for c in cases:
            print(f"  {c['name']:40s} [{c['category']}] {c['description']}")
        return

    if args.case:
        keyword = args.case.lower()
        cases = [c for c in cases if keyword in c["name"].lower() or keyword in c["category"].lower()]
        if not cases:
            print(f"没有匹配 {args.case!r} 的用例")
            return

    results = []
    for case in cases:
        print(f"▶ 运行: {case['name']} ...")
        try:
            results.append(await run_single_case(case, use_real_llm=args.real))
        except TimeoutError as e:
            print(f"    ⚠ {e}")
            results.append({
                "name": case["name"],
                "category": case.get("category", ""),
                "description": case.get("description", ""),
                "mode": "real" if args.real else "mock",
                "assertions": [{"desc": "超时未完成", "passed": False, "detail": str(e)}],
                "outbound_count": 0,
                "llm_calls_total": 0,
                "log_path": "",
            })
        except Exception as e:  # noqa: BLE001
            print(f"    ✗ 运行异常: {e}")
            results.append({
                "name": case["name"],
                "category": case.get("category", ""),
                "description": case.get("description", ""),
                "mode": "real" if args.real else "mock",
                "assertions": [{"desc": "运行异常", "passed": False, "detail": str(e)}],
                "outbound_count": 0,
                "llm_calls_total": 0,
                "log_path": "",
            })

    report = render_report(results)
    print()
    print(report)

    report_path = Path(__file__).resolve().parent / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存到: {report_path}")

    any_fail = any(not all(a["passed"] for a in r["assertions"]) for r in results)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
