"""A/B experiment for eager tool schemas versus tool_search deferred loading.

The experiment runs the production ``agent.looping.core.AgentLoop`` with
deterministic, side-effect-free probe tools.  It has two modes:

* deterministic: a schema-aware scripted provider validates the harness and
  measures structural overhead without spending model credits;
* real: the configured LLM decides which tools to call, producing an actual
  capability measurement.  Run a case more than once because model behavior
  is probabilistic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from agent.bus.event_bus import EventBus
from agent.bus.queue import MessageBus
from agent.config_models import AgentConfig
from agent.llm.client import LLMClient
from agent.looping.core import AgentLoop
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig as LoopLLMConfig,
)
from agent.provider import LLMProvider, LLMResponse, ToolCall
from agent.tools.base import BaseTool, ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from session.manager import SessionManager


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "tool_discovery_ab.json"
RESULTS_DIR = ROOT / "results"
TMP_DIR = ROOT / ".tmp_tool_discovery_ab"


@dataclass
class CaseResult:
    variant: str
    run: int
    case_id: str
    category: str
    success: bool
    expected_tools: list[str]
    actual_tools: list[str]
    missing_tools: list[str]
    forbidden_tools_called: list[str]
    order_ok: bool
    state_ok: bool
    tool_search_called: bool
    expected_tools_loaded: bool
    llm_calls: int
    tool_calls: int
    unnecessary_tool_calls: int
    initial_schema_count: int
    initial_schema_bytes: int
    total_schema_bytes: int
    latency_ms: float
    reply: str
    error: str = ""


class ProbeTool(BaseTool):
    """A deterministic tool with production-like schema and recorded calls."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        recorder: dict[str, Any],
        handler: Callable[[dict[str, Any]], str],
    ) -> None:
        self.name = name
        self.description = description
        self._parameters = parameters
        self._recorder = recorder
        self._handler = handler

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        self._recorder["tool_calls"].append(
            {"name": self.name, "arguments": dict(kwargs)}
        )
        return self._handler(dict(kwargs))


def _object_schema(
    properties: dict[str, dict[str, Any]], required: list[str]
) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def _build_probe_registry(
    recorder: dict[str, Any],
    state: dict[str, Any],
    *,
    deferred: bool,
) -> ToolRegistry:
    registry = ToolRegistry()

    def register(
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any]], str],
        *,
        risk: str = "read-only",
        aliases: tuple[str, ...] = (),
    ) -> None:
        registry.register(
            ProbeTool(
                name=name,
                description=description,
                parameters=parameters,
                recorder=recorder,
                handler=handler,
            ),
            risk=risk,
            aliases=aliases,
            source_type="evaluation-probe",
        )

    def read_file(args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        return state["files"].get(path, f"文件不存在：{path}")

    def list_dir(_: dict[str, Any]) -> str:
        return "src/\nsrc/app.py\nsrc/calculator.py\ntests/\ntests/test_calculator.py"

    def edit_file(args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        content = state["files"].get(path)
        if content is None:
            return f"修改失败：文件不存在 {path}"
        if not old_text or content.count(old_text) != 1:
            return "修改失败：old_text 不存在或不唯一"
        state["files"][path] = content.replace(old_text, new_text, 1)
        return f"已修改 {path}\n- {old_text}\n+ {new_text}"

    def write_file(args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        state["files"][path] = str(args.get("content", ""))
        return f"已写入 {path}"

    def check_syntax(args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        if path not in state["files"]:
            return f"Python 语法检查失败：文件不存在 {path}"
        return f"Python 语法检查通过：{path}"

    def run_tests(_: dict[str, Any]) -> str:
        fixed = "return left + right" in state["files"]["src/calculator.py"]
        state["tests_passed"] = fixed
        return "pytest 状态：通过\n1 passed" if fixed else (
            "pytest 状态：失败\nassert -1 == 5"
        )

    path_property = {"path": {"type": "string", "description": "workspace 内路径"}}
    register(
        "read_file",
        "读取 workspace 内的文本文件或 Python 代码，不修改内容。",
        _object_schema(path_property, ["path"]),
        read_file,
        aliases=("读取文件", "阅读代码", "查看代码"),
    )
    register(
        "list_dir",
        "列出 workspace 内的项目目录和文件。",
        _object_schema(path_property, []),
        list_dir,
        aliases=("查看目录", "项目结构", "文件列表"),
    )
    register(
        "edit_file",
        "用 old_text 到 new_text 的精确替换修改 workspace 内代码。",
        _object_schema(
            {
                **path_property,
                "old_text": {"type": "string", "description": "要被替换的原文"},
                "new_text": {"type": "string", "description": "替换后的新文本"},
            },
            ["path", "old_text", "new_text"],
        ),
        edit_file,
        risk="write",
        aliases=("修改文件", "修改代码", "编辑代码"),
    )
    register(
        "write_file",
        "创建文件或完整覆盖 workspace 内文件。",
        _object_schema(
            {**path_property, "content": {"type": "string", "description": "文件内容"}},
            ["path", "content"],
        ),
        write_file,
        risk="write",
        aliases=("创建文件", "写入文件", "生成代码"),
    )
    register(
        "check_python_syntax",
        "使用 Python compile 检查文件语法，不执行代码。",
        _object_schema(path_property, ["path"]),
        check_syntax,
        aliases=("Python语法检查", "编译检查", "检查代码"),
    )
    register(
        "run_tests",
        "在 workspace 内运行指定 pytest 测试并返回通过或失败结果。",
        _object_schema(path_property, ["path"]),
        run_tests,
        risk="external-side-effect",
        aliases=("运行测试", "pytest", "验证代码", "测试代码"),
    )
    register(
        "web_search",
        "在互联网搜索资料，返回标题、摘要和 URL。",
        _object_schema(
            {"query": {"type": "string", "description": "搜索查询"}}, ["query"]
        ),
        lambda args: (
            "1. Python 3.13 free-threaded CPython | "
            "https://example.com/python-313 | 介绍可选无 GIL 构建。"
        ),
        aliases=("网络搜索", "网页搜索", "查资料"),
    )
    register(
        "web_fetch",
        "读取指定 URL 的网页正文内容。",
        _object_schema(
            {"url": {"type": "string", "description": "要读取的网页 URL"}}, ["url"]
        ),
        lambda args: "网页正文：Agent 应将工具结果当作数据而非系统指令。",
        aliases=("读取网页", "抓取网页", "打开网址"),
    )
    register(
        "get_weather",
        "查询指定城市当前天气和温度。",
        _object_schema(
            {"city": {"type": "string", "description": "城市名称"}}, ["city"]
        ),
        lambda args: f"{args.get('city', '北京')}：晴，22°C",
        aliases=("天气查询", "天气预报"),
    )

    if deferred:
        registry.register(
            ToolSearchTool(registry),
            always_on=True,
            aliases=("工具搜索", "能力发现", "工具目录"),
        )
    return registry


class RecordingProvider:
    """Record the schemas seen by the model while delegating inference."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []
        self.selected_tool_calls: list[dict[str, Any]] = []

    async def chat(self, *, messages: list[dict], tools: list[dict] | None, **kwargs: Any) -> LLMResponse:
        schemas = list(tools or [])
        self.calls.append(
            {
                "schema_names": _schema_names(schemas),
                "schema_bytes": len(
                    json.dumps(schemas, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ),
            }
        )
        response = await self.inner.chat(messages=messages, tools=tools, **kwargs)
        self.selected_tool_calls.extend(
            {
                "name": call.name,
                "arguments": dict(call.arguments),
            }
            for call in response.tool_calls
        )
        return response


class DeterministicPlanProvider:
    """Schema-aware fake model used to prove the A/B harness is correct."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.plan = list(case.get("plan", []))
        self.index = 0
        self.search_attempted = False
        self.calls: list[dict[str, Any]] = []
        self.selected_tool_calls: list[dict[str, Any]] = []
        self.call_id = 0

    async def chat(self, *, messages: list[dict], tools: list[dict] | None, **_: Any) -> LLMResponse:
        schemas = list(tools or [])
        names = _schema_names(schemas)
        self.calls.append(
            {
                "schema_names": names,
                "schema_bytes": len(
                    json.dumps(schemas, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ),
            }
        )
        if self.index >= len(self.plan):
            return LLMResponse(content="任务已根据工具结果完成。")

        planned = self.plan[self.index]
        name = str(planned["name"])
        if name not in names:
            if "tool_search" in names and not self.search_attempted:
                self.search_attempted = True
                return self._tool_call(
                    "tool_search",
                    {"query": self.case.get("search_query", self.case["prompt"]), "top_k": 20},
                )
            return LLMResponse(content=f"未能加载所需工具 {name}。")

        self.index += 1
        return self._tool_call(name, dict(planned.get("arguments", {})))

    def _tool_call(self, name: str, arguments: dict[str, Any]) -> LLMResponse:
        self.call_id += 1
        self.selected_tool_calls.append(
            {"name": name, "arguments": dict(arguments)}
        )
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"eval_call_{self.call_id}",
                    name=name,
                    arguments=arguments,
                )
            ],
        )


def _schema_names(schemas: list[dict[str, Any]]) -> list[str]:
    return [
        str(schema.get("function", {}).get("name", ""))
        for schema in schemas
        if schema.get("function", {}).get("name")
    ]


def load_dataset(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset.get("cases"), list) or not dataset["cases"]:
        raise ValueError(f"评测集没有 cases：{path}")
    return dataset


def _is_subsequence(required: list[str], actual: list[str]) -> bool:
    cursor = 0
    for name in actual:
        if cursor < len(required) and name == required[cursor]:
            cursor += 1
    return cursor == len(required)


def _evaluate_state(case: dict[str, Any], state: dict[str, Any]) -> bool:
    expected = case.get("state_assertions") or {}
    if not expected:
        return True
    path = expected.get("file")
    contains = expected.get("contains")
    if path and contains and contains not in state["files"].get(path, ""):
        return False
    if "tests_passed" in expected:
        return state.get("tests_passed") == bool(expected["tests_passed"])
    return True


async def _run_case(
    case: dict[str, Any],
    *,
    variant: str,
    run_number: int,
    mode: str,
    real_client: LLMClient | None,
    model: str,
    system_prompt: str,
    max_iterations: int,
) -> CaseResult:
    deferred = variant == "deferred"
    recorder: dict[str, Any] = {"tool_calls": []}
    state: dict[str, Any] = {
        "files": {
            "src/calculator.py": (
                "def add(left, right):\n    return left - right\n"
            ),
            "src/app.py": "def main():\n    return 'ok'\n",
            "tests/test_calculator.py": (
                "from src.calculator import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
            ),
        },
        "tests_passed": False,
    }
    tools = _build_probe_registry(recorder, state, deferred=deferred)
    if mode == "real":
        if real_client is None:
            raise ValueError("真实模式缺少 LLM client")
        provider: Any = RecordingProvider(LLMProvider(real_client))
    else:
        provider = DeterministicPlanProvider(case)

    run_dir = TMP_DIR / f"{variant}-{case['id']}-{run_number}-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    bus = MessageBus()
    events = EventBus()
    sessions = SessionManager(run_dir)
    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            event_bus=events,
            provider=provider,
            tools=tools,
            session_manager=sessions,
            workspace=run_dir,
            memory_service=None,
        ),
        AgentLoopConfig(
            llm=LoopLLMConfig(
                model=model,
                max_iterations=max_iterations,
                max_tokens=2048,
                tool_timeout_seconds=5.0,
                tool_search_enabled=deferred,
                system_prompt=(
                    system_prompt.strip()
                    + "\n这是受控评测任务。需要外部信息时必须真正调用当前可用工具，"
                    "不得假装已经读取、搜索、修改或测试。"
                ),
            )
        ),
    )

    started = time.perf_counter()
    reply = ""
    error = ""
    try:
        reply = await loop.process_direct(
            case["prompt"],
            session_key=f"eval:{variant}:{case['id']}:{run_number}",
            channel="eval",
            chat_id=case["id"],
        )
    except Exception as exc:  # keep the remaining benchmark cases running
        error = f"{type(exc).__name__}: {exc}"
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        sessions.close()
        await events.close()
        shutil.rmtree(run_dir, ignore_errors=True)

    # Provider-side records include meta tools such as tool_search; probe-side
    # records only include the functional tool implementations.
    calls = list(provider.selected_tool_calls)
    actual_tools = [str(call["name"]) for call in calls]
    expected_tools = [str(name) for name in case.get("expected_tools", [])]
    missing = [name for name in expected_tools if name not in actual_tools]
    forbidden = [
        str(name)
        for name in case.get("forbidden_tools", [])
        if name in actual_tools
    ]
    required_order = [str(name) for name in case.get("required_order", [])]
    order_ok = not required_order or _is_subsequence(required_order, actual_tools)
    state_ok = _evaluate_state(case, state)
    functional_calls = [name for name in actual_tools if name != "tool_search"]
    useful_once = len(set(functional_calls).intersection(expected_tools))
    unnecessary = max(0, len(functional_calls) - useful_once)
    snapshots = list(provider.calls)
    union_schemas = {
        name for snapshot in snapshots for name in snapshot["schema_names"]
    }
    loaded = all(name in union_schemas for name in expected_tools)
    success = not error and not missing and not forbidden and order_ok and state_ok

    return CaseResult(
        variant=variant,
        run=run_number,
        case_id=str(case["id"]),
        category=str(case.get("category", "")),
        success=success,
        expected_tools=expected_tools,
        actual_tools=actual_tools,
        missing_tools=missing,
        forbidden_tools_called=forbidden,
        order_ok=order_ok,
        state_ok=state_ok,
        tool_search_called="tool_search" in actual_tools,
        expected_tools_loaded=loaded,
        llm_calls=len(snapshots),
        tool_calls=len(actual_tools),
        unnecessary_tool_calls=unnecessary,
        initial_schema_count=(len(snapshots[0]["schema_names"]) if snapshots else 0),
        initial_schema_bytes=(int(snapshots[0]["schema_bytes"]) if snapshots else 0),
        total_schema_bytes=sum(int(item["schema_bytes"]) for item in snapshots),
        latency_ms=round(latency_ms, 2),
        reply=reply,
        error=error,
    )


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    attempted = len(results)
    valid_results = [item for item in results if not item.error]
    total = len(valid_results)
    functional_calls = sum(
        sum(1 for name in item.actual_tools if name != "tool_search")
        for item in valid_results
    )
    expected_total = sum(len(item.expected_tools) for item in valid_results)
    expected_matched = sum(
        len(item.expected_tools) - len(item.missing_tools) for item in valid_results
    )
    unique_selected = sum(
        len(set(name for name in item.actual_tools if name != "tool_search"))
        for item in valid_results
    )
    useful_selected = sum(
        len(
            set(name for name in item.actual_tools if name != "tool_search")
            .intersection(item.expected_tools)
        )
        for item in valid_results
    )
    return {
        "attempted_runs": attempted,
        "valid_runs": total,
        "infrastructure_error_rate": round(
            (attempted - total) / attempted if attempted else 0.0, 4
        ),
        "task_success_rate": round(
            sum(item.success for item in valid_results) / total if total else 0.0, 4
        ),
        "tool_selection_recall": round(
            expected_matched / expected_total if expected_total else 0.0, 4
        ),
        "tool_selection_precision": round(
            useful_selected / unique_selected if unique_selected else 0.0, 4
        ),
        "schema_load_rate": round(
            sum(item.expected_tools_loaded for item in valid_results) / total if total else 0.0,
            4,
        ),
        "dynamic_discovery_rate": round(
            sum(item.tool_search_called for item in valid_results) / total if total else 0.0,
            4,
        ),
        "safety_violation_rate": round(
            sum(bool(item.forbidden_tools_called) for item in valid_results) / total
            if total
            else 0.0,
            4,
        ),
        "unnecessary_tool_call_rate": round(
            sum(item.unnecessary_tool_calls for item in valid_results) / functional_calls
            if functional_calls
            else 0.0,
            4,
        ),
        "avg_initial_schema_count": _mean(
            [float(item.initial_schema_count) for item in valid_results]
        ),
        "avg_initial_schema_bytes": _mean(
            [float(item.initial_schema_bytes) for item in valid_results]
        ),
        "avg_total_schema_bytes": _mean(
            [float(item.total_schema_bytes) for item in valid_results]
        ),
        "avg_llm_calls": _mean([float(item.llm_calls) for item in valid_results]),
        "avg_tool_calls": _mean([float(item.tool_calls) for item in valid_results]),
        "avg_latency_ms": _mean([item.latency_ms for item in valid_results]),
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_markdown(payload: dict[str, Any]) -> str:
    aggregates = payload["aggregates"]
    baseline = aggregates.get("baseline")
    deferred = aggregates.get("deferred")
    lines = [
        "# tool_search + 延迟工具加载 A/B 评测报告",
        "",
        f"- 模式：`{payload['mode']}`",
        f"- 模型：`{payload['model']}`",
        f"- 数据集：`{payload['dataset']}`",
        f"- 每个 case 重复：`{payload['runs_per_case']}` 次",
        f"- 生成时间：`{payload['generated_at']}`",
        "",
    ]
    if payload["mode"] == "deterministic":
        lines.extend(
            [
                "> 确定性模式只验证评测器、Agent Loop 和延迟加载机制，",
                "> 不代表真实大模型的工具选择能力。面试数据应使用 `--mode real` 结果。",
                "",
            ]
        )
    if baseline and deferred:
        metrics = [
            ("基础设施错误率", "infrastructure_error_rate", "percent"),
            ("任务成功率", "task_success_rate", "percent"),
            ("工具选择召回率", "tool_selection_recall", "percent"),
            ("工具选择准确率", "tool_selection_precision", "percent"),
            ("所需 Schema 加载率", "schema_load_rate", "percent"),
            ("动态搜索调用率", "dynamic_discovery_rate", "percent"),
            ("安全违规率", "safety_violation_rate", "percent"),
            ("无效/重复工具调用率", "unnecessary_tool_call_rate", "percent"),
            ("平均初始 Schema 数", "avg_initial_schema_count", "number"),
            ("平均初始 Schema 字节", "avg_initial_schema_bytes", "number"),
            ("平均累计 Schema 字节", "avg_total_schema_bytes", "number"),
            ("平均 LLM 调用数", "avg_llm_calls", "number"),
            ("平均工具调用数", "avg_tool_calls", "number"),
            ("平均耗时（ms）", "avg_latency_ms", "number"),
        ]
        lines.extend(
            [
                "## 汇总对比",
                "",
                "| 指标 | A：全量 Schema | B：tool_search 延迟加载 | B - A |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, key, kind in metrics:
            left = float(baseline[key])
            right = float(deferred[key])
            delta = right - left
            if kind == "percent":
                values = (_format_percent(left), _format_percent(right), f"{delta * 100:+.1f}pp")
            else:
                values = (f"{left:.2f}", f"{right:.2f}", f"{delta:+.2f}")
            lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
        lines.append("")

    lines.extend(
        [
            "## 分用例结果",
            "",
            "| 版本 | 轮次 | 用例 | 成功 | 实际工具链 | 初始 Schema | LLM 调用 | 耗时 ms |",
            "|---|---:|---|:---:|---|---:|---:|---:|",
        ]
    )
    for item in payload["case_results"]:
        chain = " → ".join(item["actual_tools"]) or "（未调用）"
        lines.append(
            f"| {item['variant']} | {item['run']} | {item['case_id']} | "
            f"{'PASS' if item['success'] else 'FAIL'} | {chain} | "
            f"{item['initial_schema_count']} | {item['llm_calls']} | {item['latency_ms']:.2f} |"
        )
        if item["error"]:
            lines.append(f"\n> `{item['case_id']}` 运行错误：{item['error']}\n")

    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "- 本实验使用与生产工具同名、同类 Schema 的无副作用探针，隔离网络波动和真实文件修改。",
            "- 评测了工具发现、选择和编排；工具实现正确性由 pytest 单元/集成测试保证。",
            "- Schema 字节数是可重现的上下文体积代理指标，不是模型服务商的真实 Token 计费数。",
            "- 真实模型具有随机性，正式数据建议每条 case 至少重复 3 次，并保留模型和数据集版本。",
            "",
        ]
    )
    return "\n".join(lines)


async def run_experiment(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    mode: str = "deterministic",
    runs: int = 1,
    variants: tuple[str, ...] = ("baseline", "deferred"),
    case_filter: str = "",
    config_path: Path = Path("config.toml"),
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    cases = list(dataset["cases"])
    if case_filter:
        lowered = case_filter.lower()
        cases = [
            case
            for case in cases
            if lowered in str(case["id"]).lower()
            or lowered in str(case.get("category", "")).lower()
        ]
    if not cases:
        raise ValueError(f"没有匹配的评测用例：{case_filter}")

    real_client: LLMClient | None = None
    model = "deterministic-plan-provider"
    system_prompt = "You are a helpful assistant."
    max_iterations = 10
    if mode == "real":
        config = AgentConfig.load(config_path)
        profile = config.llm.main
        if not profile.api_key:
            raise ValueError(f"真实模式缺少 API Key：{config_path}")
        model = profile.model
        system_prompt = config.system_prompt
        max_iterations = config.max_iterations
        real_client = LLMClient(
            api_key=profile.api_key,
            base_url=profile.base_url,
            model=profile.model,
        )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    try:
        for variant in variants:
            for run_number in range(1, runs + 1):
                for case in cases:
                    print(
                        f"RUN {variant} run={run_number} case={case['id']}",
                        flush=True,
                    )
                    results.append(
                        await _run_case(
                            case,
                            variant=variant,
                            run_number=run_number,
                            mode=mode,
                            real_client=real_client,
                            model=model,
                            system_prompt=system_prompt,
                            max_iterations=max_iterations,
                        )
                    )
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    aggregates = {
        variant: aggregate([item for item in results if item.variant == variant])
        for variant in variants
    }
    return {
        "experiment": "tool_discovery_ab",
        "dataset": str(dataset_path),
        "dataset_name": dataset.get("name", ""),
        "mode": mode,
        "model": model,
        "runs_per_case": runs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variants": list(variants),
        "aggregates": aggregates,
        "case_results": [asdict(item) for item in results],
    }


def save_report(payload: dict[str, Any]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = str(payload["mode"])
    json_path = RESULTS_DIR / f"tool_discovery_ab_{suffix}.json"
    markdown_path = RESULTS_DIR / f"tool_discovery_ab_{suffix}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return markdown_path, json_path


def _parse_variants(raw: str) -> tuple[str, ...]:
    if raw == "both":
        return ("baseline", "deferred")
    return (raw,)


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="对比全量工具 Schema 与 tool_search 延迟加载"
    )
    parser.add_argument(
        "--mode", choices=("deterministic", "real"), default="deterministic"
    )
    parser.add_argument("--runs", type=int, default=1, help="每个 case 重复次数")
    parser.add_argument(
        "--variant",
        choices=("baseline", "deferred", "both"),
        default="both",
    )
    parser.add_argument("--case", default="", help="按 id 或 category 过滤")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必须大于等于 1")

    payload = await run_experiment(
        dataset_path=args.dataset,
        mode=args.mode,
        runs=args.runs,
        variants=_parse_variants(args.variant),
        case_filter=args.case,
        config_path=args.config,
    )
    markdown_path, json_path = save_report(payload)
    print("\n" + _render_markdown(payload))
    print(f"\nMarkdown 报告：{markdown_path}")
    print(f"JSON 报告：{json_path}")


if __name__ == "__main__":
    asyncio.run(main())

