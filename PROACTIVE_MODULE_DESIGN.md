# Akashic Proactive 模块完整设计文档

> 基于 `akashic-agent-main/proactive_v2/` 源码分析
> 版本：v2 架构
> 更新：2025

---

## 📋 目录

1. [整体架构概览](#整体架构概览)
2. [核心入口：ProactiveLoop](#核心入口proactueloop)
3. [调度器与电量模型](#调度器与电量模型)
4. [8 Phase 编排机制](#8-phase-编排机制)
5. [核心 Pipeline：ProactiveTurnPipeline](#核心-pipelineproactiveturnpipeline)
6. [Gate 阶段：5 道准入门](#gate-阶段5-道准入门)
7. [Source 阶段：数据拉取](#source-阶段数据拉取)
8. [Judge 阶段：LLM 多轮工具循环](#judge-阶段llm-多轮工具循环)
9. [Resolve 阶段：去重/决策/ACK](#resolve-阶段去重决策ack)
10. [Deliver 阶段：发送+ACK+日志](#deliver-阶段发送ack日志)
11. [Drift 空闲任务](#drift-空闲任务)
12. [配置全景](#配置全景)
12. [关键数据结构](#关键数据结构)
13. [关键设计模式总结](#关键设计模式总结)

---

## 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ProactiveLoop.run()                          │
├─────────────────────────────────────────────────────────────────────┤
│  ProactiveScheduler.next_interval()  ──→  自适应 tick 间隔          │
│  ProactiveKernel.run_tick()                                          │
│     └─ ProactivePhaseRunner (8 Phase 串行执行)                      │
│        ├─ proactive.tick        (tick 入口)                         │
│        ├─ proactive.gate        (准入检查)                          │
│        ├─ proactive.source      (MCP 连接池管理)                    │
│        ├─ proactive.drift       (Drift 空闲任务分支)                 │
│        ├─ proactive.prompt      (Prompt 构建)                       │
│        ├─ proactive.judge       (LLM 多轮工具循环)                   │
│        ├─ proactive.resolve     (去重/决策)                         │
│        └─ proactive.deliver     (发送 + ACK + 日志)                 │
└─────────────────────────────────────────────────────────────────────┘
```

**核心文件映射**：
| 文件 | 核心类 | 职责 |
|------|--------|------|
| `loop.py` | `ProactiveLoop` | 主循环、调度、tick 执行 |
| `phases.py` | `ProactivePhaseRunner` | 8 Phase 编排器 |
| `frame.py` | `ProactiveFrame` | Phase 间数据流载体 |
| `context.py` | `AgentTickContext` | tick 级上下文 |
| `modules_gate.py` | `ProactiveGateChain` | 5 道准入门 |
| `modules_source.py` | `DataGateway` / `McpRuntimeModule` | 数据拉取 + MCP 连接池 |
| `modules_prompt.py` | `ProactivePromptBuilder` | Prompt 构建 |
| `modules_judge.py` | `ProactiveJudge` | LLM 多轮工具循环 |
| `modules_resolve.py` | `ProactiveResolver` | 去重/决策/ACK |
| `modules_deliver.py` | `ProactiveDeliverer` | 发送 + ACK + 日志 |
| `agent/core/proactive_turn.py` | `ProactiveTurnPipeline` | 5 段 Pipeline 编排 |
| `scheduler.py` | `ProactiveScheduler` | 调度器、电量模型 |
| `energy.py` | `EnergyModel` | 电量模型 |

---

## 核心入口：ProactiveLoop

**文件**：`proactive_v2/loop.py` - `ProactiveLoop` 类

### 生命周期

```python
async def run(self):
    self._running = True
    await self._proactive_kernel.start()
    try:
        await self._run_loop()
    finally:
        await self._proactive_kernel.stop()

async def _run_loop(self):
    last_base_score = None
    while self._running:
        interval = self._next_interval(last_base_score)
        await asyncio.sleep(interval)
        try:
            last_base_score = await self._tick()
        except Exception:
            logger.exception("ProactiveLoop tick 异常")
            last_base_score = None

async def _tick(self):
    _ = current_session_key.set(self._target_session_key())
    score = await self._proactive_kernel.run_tick(session_key)
    return score
```

**关键点**：
- `_next_interval(base_score)` → 调度器根据上一轮 `base_score` 计算下次间隔
- `_tick()` 调用 `kernel.run_tick()` 执行完整 Phase 链
- 返回 `base_score` 给调度器用于下次间隔计算

### 组件装配 (`_init_runtime_components`)

```python
def _init_runtime_components(self):
    # 1. PROACTIVE_CONTEXT.md 规则面板
    self._ensure_workspace_proactive_context_file()
    self._read_workspace_proactive_context()
    
    # 2. 核心组件装配
    self._turn_orchestrator = self._build_turn_orchestrator()
    self._sense = self._build_sense()
    self._mcp_runtime = self._build_mcp_runtime()
    self._scheduler = ProactiveScheduler(...)
    self._proactive_kernel = self._build_kernel()
```

---

## 调度器与电量模型

### EnergyModel (`energy.py`)

```python
class EnergyModel:
    def __init__(self, half_life_hours: float = 4.0):
        self.half_life = half_life_hours * 3600  # 秒
    
    def energy(self, last_user_at, now):
        if last_user_at is None: return 0.0
        elapsed = (now - last_user_at).total_seconds()
        return 0.5 ** (elapsed / self.half_life)  # 指数衰减
```

**电量衰减曲线**：
```
时间 →     刚聊完      4h后        8h后         24h后
energy →   1.0         0.5         0.25         ~0
d_energy →  0           0.5         0.75         ~1
interval →   8min        4min         4min        4min
状态    →    s0(不饿)    s1(饿)        s1(饿)        s1(饿)
```

### ProactiveScheduler

```python
class ProactiveScheduler:
    def next_interval(self, base_score=None):
        energy = self._energy_model.energy(last_user_at, now)
        d_energy = 1.0 - energy  # 饥渴度
        base_score = d_energy
        
        if base_score <= threshold:  # s0: 不饿
            interval = tick_interval_s0  # 8min
        else:                          # s1: 饿
            interval = tick_interval_s1  # 4min
        
        jitter = rng.uniform(-tick_jitter, tick_jitter)
        return max(1, int(interval + jitter))
```

**三档预设** (config.toml)：
| 预设 | s0间隔 | s1间隔 | 半衰期 | 适用场景 |
|------|--------|--------|--------|----------|
| `daily` | 8min | 4min | 4h | 日常 |
| `dev_verify` | 30s | 10s | 0.5h | 调试 |
| `quiet` | 30min | 15min | 8h | 低打扰 |

---

## 8 Phase 编排机制

### Phase 定义 (`phases.py`)

```python
PROACTIVE_PHASES = (
    "proactive.tick",      # 入口
    "proactive.gate",      # 准入检查
    "proactive.source",    # MCP 连接池管理
    "proactive.drift",     # Drift 空闲任务分支
    "proactive.prompt",    # Prompt 构建
    "proactive.judge",     # LLM 多轮工具循环
    "proactive.resolve",   # 去重/决策
    "proactive.deliver",   # 发送 + ACK + 日志
)

class ProactivePhaseRunner:
    def __init__(self, modules):
        # 1. 按 phase 分组
        for module in modules:
            phase = getattr(module, "phase", None)
            grouped[phase].append(module)
        
        # 2. 拓扑排序 + 循环依赖检测
        self._modules_by_phase = {
            phase: topo_sort_modules(grouped.get(phase, []))
            for phase in PROACTIVE_PHASES
        }
    
    async def run(self, frame):
        for phase in PROACTIVE_PHASES:
            for module in self._modules_by_phase[phase]:
                frame = await module.run(frame)
        return frame
```

**关键设计**：
- 模块声明 `phase = "proactive.xxx"` + `slot` + `requires/produces`
- 启动时自动拓扑排序、循环依赖检测、缺失依赖自动禁用
- `Frame` 在 Phase 间传递，`slots` 存中间数据

---

## 核心 Pipeline：ProactiveTurnPipeline

**文件**：`agent/core/proactive_turn.py` (636 行)

### 类结构

```python
class ProactiveTurnPipeline:
    slot = "proactive.tick.pipeline"
    phase = "proactive.deliver"
    
    def __init__(self, deps: ProactiveTurnPipelineDeps):
        # 1. 保存配置和核心依赖
        # 2. 初始化 5 大核心组件
        self._gate_chain = ProactiveGateChain(...)
        self._prompt_builder = ProactivePromptBuilder(...)
        self._judge = ProactiveJudge(...)
        self._resolver = ProactiveResolver(...)
        self._deliverer = ProactiveDeliverer(...)
        
        # Drift pipeline 共享 step_recorder
        if self._drift_pipeline and self._drift_pipeline.step_recorder is None:
            self._drift_pipeline.step_recorder = self._record_tick_step
```

### 核心入口：`run()` (5 段流程)

```python
async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
    # 1. 创建上下文
    ctx = AgentTickContext(session_key, started_at)
    
    # 1. Gate — 该不该动？
    gate = self._gate_check(ctx)
    if gate.blocked:
        return frame.with_output(base_score=gate.base_score)
    
    ctx.context_as_fallback_open = gate.context_as_fallback_open
    
    # 2. Fetch — 数据拉取
    feed = await self._fetch_pull(ctx)
    if feed.drift_entered:
        return feed.base_score
    
    # 3. Judge — LLM 评估
    if feed.messages and not ctx.terminal_action:
        await self._judge_evaluate(ctx, feed.messages)
    
    # 4. Resolve — 决策去重
    decision = await self._resolve_decide(ctx)
    
    # 5. Deliver — 执行发送
    score = await self._deliver_execute(ctx, decision)
    
    return frame
```

---

## Gate 阶段：5 道准入门

**文件**：`modules_gate.py` - `ProactiveGateChain`

```python
def check(self, ctx) -> GateResult:
    # 1. 无目标
    if not self._cfg.default_chat_id:
        return GateResult(blocked=True, reason="no_target")
    
    # 2. 被动忙碌
    if self._passive_busy_fn and self._passive_busy_fn(self._session_key):
        return GateResult(blocked=True, reason="busy")
    
    # 2. 发送冷却
    if self._state_store.count_deliveries_in_window(...) > 0:
        return GateResult(blocked=True, reason="cooldown")
    
    # 3. AnyAction Gate (电量模型)
    if self._any_action_gate:
        should_act, meta = self._any_action_gate.should_act(now_utc, last_user_at)
        if not should_act:
            return GateResult(blocked=True, reason="presence")
    
    # 4. Context Fallback 开关
    return GateResult(
        blocked=False,
        reason="passed",
        context_as_fallback_open=self._context_as_fallback_open(ctx)
    )
```

**Context Fallback 开关逻辑** (`_context_as_fallback_open`)：
```python
def _context_as_fallback_open(self, ctx):
    # 概率阀
    if self._rng.random() >= self._cfg.agent_tick_context_prob:
        return False
    # 频控：24h 内最多 N 次
    last_at = self._state_store.get_last_context_only_at(...)
    count_24h = self._state_store.count_context_only_in_window(...)
    if last_at and elapsed < min_interval_hours * 3600:
        return False
    return count_24h < self._cfg.context_only_daily_max
```

---

## Source 阶段：数据拉取

### DataGateway 并行拉取 (`modules_source.py`)

```python
class DataGateway:
    async def run(self) -> GatewayResult:
        # 并行拉取三路
        alerts_task = asyncio.create_task(self._fetch_alerts())
        contents_task = asyncio.create_task(self._fetch_contents())
        context_task = asyncio.create_task(self._fetch_context())
        
        alerts, contents, context = await asyncio.gather(...)
        
        # 内容预取：并行 web_fetch 取正文
        if content_meta:
            await self._prefetch_contents(gw_result.content_meta)
        
        return GatewayResult(alerts, content_meta, content_store, context)
```

**关键优化**：
- 并行拉取 alert/content/context
- 内容预取：并行 `web_fetch` 取正文 → 存入 `content_store` → Judge 直接读取

### McpRuntimeModule - 连接池管理

```python
class McpRuntimeModule:
    async def start(self):
        await self._pool.connect_all()
        await self._poll_once()
        self._poll_task = asyncio.create_task(self._poll_loop())
    
    async def _poll_loop(self):
        while self._running:
            await asyncio.sleep(cfg.feed_poller_interval_seconds)
            await self._poll_once()  # 后台拉取 feed 内容
```

---

## Judge 阶段：LLM 多轮工具循环

**文件**：`modules_judge.py` - `ProactiveJudge`

```python
async def evaluate(self, ctx, messages, gw_result):
    while ctx.steps_taken < cfg.agent_tick_max_steps:
        ok = await self._run_tool_step(messages, ctx, tool_choice="auto")
        if not ok: break
        if ctx.terminal_action: break
    
    # 完整性检查：未分类项强制补全
    if ctx.terminal_action == "skip" and gw_result.content_meta:
        unclassified = all_ids - (interesting | discarded)
        if unclassified:
            messages.append({"role": "user", "content": completeness_msg})
            for _ in range(5):  # 最多 5 轮补全
                ok = await _run_tool_step(messages, ctx, loop_tag="complete")
                if not ok: break
    
    # 反思收尾
    if ctx.terminal_action is None and ctx.interesting_item_ids:
        reflection = f"已标记 interesting: {ids_str}，必须 message_push + finish_turn(reply)"
        messages.append({"role": "user", "content": reflection})
        for _ in range(3): ...

async def _run_tool_step(self, messages, ctx, tool_choice="auto"):
    response = await llm_fn(messages, tools=active_schemas, tool_choice)
    
    if not response.has_tool_calls:
        ctx.terminal_action = "reply"
        return True
    
    # 执行工具
    for tc in response.tool_calls:
        name = tc.function.name
        args = _parse_arguments(tc.function.arguments)
        _trace_tool_start(name)
        tool_result = await _run_tool(tools, name, **args)
        _trace_tool_end(name)
        messages.append({"role": "tool", "content": tool_result, "tool_call_id": tc.id})
    
    return True
```

**关键机制**：
1. **工具循环**：LLM 返回 tool_calls → 执行 → 结果回填 messages → 继续循环
2. **完整性检查** - skip 时检查是否有未分类项，强制补全
3. **反思收尾** - 有 interesting 但未 reply → 强制注入反思提示词
4. **步数限制** - `agent_tick_max_steps` (默认 20) 防死循环

---

## Resolve 阶段：去重/决策/ACK

**文件**：`modules_resolve.py` - `ProactiveResolver`

```python
async def resolve(self, ctx) -> ResolveResult:
    if ctx.terminal_action != "reply":
        return self._build_skip_result(ctx)
    
    delivery_key = build_delivery_key(ctx)
    
    # 1. Delivery 去重
    if state_store.is_delivery_duplicate(session_key, delivery_key, hours):
        return _build_delivery_dedupe_result(ctx)
    
    # 2. Message 去重
    if message_dedupe_enabled and deduper:
        is_dup, reason = await deduper.is_duplicate(...)
        if is_dup:
            return _build_message_dedupe_result(ctx, reason)
    
    # 3. 发送
    return _build_send_result(ctx, delivery_key)
```

### 三层去重

| 层级 | Key 构建 | TTL | 说明 |
|------|----------|-----|------|
| **Delivery Dedup** | `sha1(json.dumps(cited_item_ids + url/title))` | `delivery_dedupe_hours` (24h) | 同一批内容不重复推 |
| **Message Dedup** | `MessageDeduper` (embedding 相似度) | `message_dedupe_recent_n` | 语义去重 |
| **ACK TTL** | cited: 168h / unreferenced: 24h / discarded: 720h | - | 不同处理结果不同 TTL |

### Delivery Key 构建

```python
def build_delivery_key(ctx):
    refs = _build_delivery_refs(ctx)  # url:/title:/id: 三种 ref
    if refs and any(not ref.startswith("id:") for ref in refs):
        key_src = json.dumps(refs)  # 有 url/title 用语义去重
    else:
        key_src = json.dumps(sorted(ctx.cited_item_ids))
    return sha1(key_src.encode()).hexdigest()[:16]
```

### ACK TTL 策略

| 常量 | 值 | 含义 |
|------|------|------|
| `_CITED_ACK_TTL` | 168h (7天) | 被引用内容 |
| `_UNCITED_ACK_TTL` | 24h | 标记 interesting 但未引用 |
| `_POST_GUARD_ACK_TTL` | 24h | post-guard 保护 |
| `_DISCARDED_ACK_TTL` | 720h (30天) | 明确丢弃 |

---

## ACK 机制完整链路

```
ProactiveResolver.ack_on_success()
        │
        ├─ cited content → ack_fn(key, 168h)
        ├─ unreferenced → ack_fn(key, 24h)
        └─ alert → alert_ack_fn()
                │
                ▼
modules_source.py:ack_fn() / alert_ack_fn()
        │
        ▼
mcp_sources.acknowledge_content_entries_async()
        │
        ▼
McpClientPool.call(server, "acknowledge_content_entries", {...})
        │
        ▼ (JSON-RPC 2.0 over stdio)
MCP Server (feed-mcp / fitbit-mcp)
        │
        ▼
DB: UPDATE items SET acknowledged=1, ack_expires_at=now()+TTL WHERE id=?
```

### MCP Server 端处理 (以 feed-mcp 为例)

```python
@mcp.tool()
async def acknowledge_content_entries(entries: list[dict], ttl_hours: int):
    for entry in entries:
        item_id = entry["id"]
        ttl_hours = entry["ttl_hours"]
        await db.execute(
            "UPDATE items SET acknowledged=1, ack_expires_at=? WHERE id=?",
            (datetime.now() + timedelta(hours=ttl_hours), item_id)
        )
    return {"status": "ok"}
```

**下次拉取时**：
```sql
SELECT * FROM items 
WHERE acknowledged=0 
  AND ack_expires_at > now()
```

---

## Deliver 阶段

```python
# modules_deliver.py
class ProactiveDeliverer:
    async def deliver(self, ctx, decision):
        self._record_finish_fn(ctx, result=decision.result)
        await self._turn_orchestrator.handle_proactive_turn(
            result=decision.result,
            session_key=self._session_key,
            channel=..., chat_id=...
        )

# TurnOrchestrator.handle_proactive_turn()
async def handle_proactive_turn(self, result, session_key, channel, chat_id):
    if result.decision == "skip":
        await self._run_side_effects(result)
        return False
    
    # 1. 持久化主动会话
    self._persist_proactive_session(session, content, media, result)
    
    # 2. 发送前副作用
    await self._run_effects(result.side_effects)
    
    # 3. 真正发送
    sent = await self._outbound.dispatch(OutboundDispatch(...))
    
    # 4. 根据结果执行副作用
    if sent:
        await self._run_effects(result.success_side_effects)
    else:
        await self._run_effects(result.failure_side_effects)
```

---

## Drift 空闲任务

**文件**：`agent/core/drift_turn.py` - `DriftTurnPipeline`

```python
class DriftTurnPipeline:
    async def run(self, ctx, llm_fn) -> bool:
        # 1. 加载技能
        skills = self._skills_loader.list_skill_records(filter_unavailable=True)
        
        # 2. 构建 system prompt + skills 描述
        system_prompt = self._build_system_prompt(skills)
        
        # 3. LLM 选择技能
        messages = [...]
        response = await llm_fn(messages, tools=skill_tools)
        
        # 3. 执行技能
        for tc in response.tool_calls:
            skill = store.get_skill(tc.function.name)
            result = await skill.execute(**args)
            messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
        
        # 4. 结束
        return True
```

**触发条件** (在 `_fetch_pull` 中)：
```python
if not gw_result.alerts and not gw_result.content_meta and not ctx.context_as_fallback_open:
    if self._drift_pipeline and self._cfg.drift_enabled:
        # 检查冷却
        last_drift_at = state_store.get_last_drift_at(...)
        if cooldown_ok:
            entered_drift = await self._drift_pipeline.run(ctx, self._llm_fn)
```

**技能定义** (`SKILL.md`)：
```markdown
---
name: audit_memory
description: 审计长期记忆是否准确
when_to_use: "定期审计，或用户投诉记忆不准时"
config:
  akashic:
    requires:
      bins: ["sqlite3"]
---
### 操作指南
1. 读取 MEMORY.md
2. 对比最近对话
3. 发现冲突 → 修正
```

---

## 配置全景 (config.toml)

```toml
[proactive]
enabled = true
profile = "daily"          # daily/quiet/dev_verify

[proactive.target]
channel = "telegram"
chat_id = "7674283004"

[proactive.agent]
max_steps = 20
content_limit = 5
web_fetch_max_chars = 8000
context_prob = 0.03

[proactive.drift]
enabled = true
max_steps = 30
min_interval_hours = 1

# 调度
tick_interval_s0 = 480      # s0: 8min
tick_interval_s1 = 240      # s1: 4min
tick_jitter = 30
energy_half_life_hours = 4.0
energy_threshold_s0 = 0.2   # s0/s1 分界线

# 去重
message_dedupe_enabled = true
message_dedupe_recent_n = 20
delivery_dedupe_hours = 24
delivery_cooldown_hours = 1

# AnyAction Gate
anyaction_enabled = true
anyaction_min_interval_seconds = 3600
anyaction_probability_min = 0.1
anyaction_probability_max = 0.5
```

---

## 关键数据结构

### AgentTickContext (context.py)

```python
@dataclass
class AgentTickContext:
    tick_id: str = uuid4().hex[:8]
    now_utc: datetime = field(default_factory=datetime.now(timezone.utc))
    session_key: str = ""
    context_as_fallback_open: bool = False
    
    # Gateway 预取结果
    fetched_alerts: list[dict] = []
    fetched_contents: list[dict] = []
    fetched_context: list[dict] = []
    content_store: dict[str, str] = {}  # url -> 正文
    
    # 过滤结果
    discarded_item_ids: set[str] = set()
    interesting_item_ids: set[str] = set()
    
    # 终止状态
    terminal_action: Literal["reply", "skip"] | None = None
    skip_reason: str = ""
    skip_note: str = ""
    final_message: str = ""
    cited_item_ids: list[str] = []
    steps_taken: int = 0
    
    # Drift
    drift_entered: bool = False
    drift_finished: bool = False
```

### ProactiveFrame (frame.py)

```python
@dataclass
class ProactiveFrame:
    input: ProactiveTickInput
    slots: dict[str, Any] = {}
    output: ProactiveTickResult | None = None

@dataclass
class ProactiveTickInput:
    session_key: str
    started_at: datetime

@dataclass
class ProactiveTickResult:
    base_score: float | None = None
```

---

## 关键设计模式总结

| 模式 | 应用场景 |
|------|----------|
| **Phase 编排** | 8 Phase 声明式声明，拓扑排序执行，插件化扩展 |
| **Factory 工厂** | `AgentTickFactory` 组装 Pipeline 依赖 |
| **Context 传递** | `AgentTickContext` 贯穿全链路，承载所有中间状态 |
| **Strategy 模式** | `ProviderStrategy` 适配不同 LLM 厂商 |
| **Callback SideEffect** | `CallbackSideEffect` 解耦副作用（ACK/日志/持久化） |
| **Event Bus** | `EventBus` 解耦发布/订阅 (observe/enqueue/fanout) |
| **State Store** | 统一状态持久化 (JSON + SQLite 索引) |
| **Diagnostic Log** | 结构化日志 + trace_id 贯穿全链路 |

---

## 调试/观测入口

```bash
# 1. 开启 Prompt 落盘
$env:PROMPT_DUMP=1; python main.py

# 2. 开启工具轨迹
$env:TOOL_TRACE=1; python main.py

# 3. 评测
python -m eval.runner --real --case 会话

# 4. 查看日志
eval/logs/session_isolation.md   # prompt 日志
tool_trace.log                    # 工具执行轨迹
eval/report.md                   # 评测报告
```

---

## 常见问题排查

| 现象 | 可能原因 | 排查方向 |
|------|----------|----------|
| 不主动推送 | `enabled=false` / `profile` 错误 / `chat_id` 为空 | 检查 config.toml、Gate 日志 |
| 重复推送 | `delivery_dedupe_hours=0` / `message_dedupe_enabled=false` | 检查 Resolver 日志、delivery_key |
| 不进入 Drift | `drift_enabled=false` / `drift_min_interval_hours` 未过 / `context_fallback` 开启 | 检查 Drift 相关配置、Gate 的 `context_as_fallback_open` |
| 电量不衰减 | `last_user_at` 未更新 / `energy_half_life_hours` 过大 | 检查 `PresenceStore.get_last_user_at` |
| MCP 连接失败 | `mcp_servers.json` 路径/命令/环境变量错误 | 检查 `mcp_servers.json`、MCP Server 日志 |

---

> 文档维护：随代码演进同步更新
> 最后更新：2025