""" Agent 核心循环 —— 消息总线的调度中枢

流程：
  consume_inbound → [记忆加载] → before_turn(门) → before_reasoning(门)
  → prompt_render(门) → [before_step(门) → LLM → after_step(观察)] × N
  → after_reasoning(门) → [记忆萃取] → [历史更新] → after_turn(观察) → publish_outbound → enqueue

工具循环：
  1. 调 LLM（附带工具 schema + 记忆上下文）
  2. LLM 返回 tool_calls → 解析并执行工具 → 结果送回 LLM → 重复
  3. LLM 返回文本 → 作为最终回复
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from agent.bus.events import InboundMessage, OutboundMessage
from agent.bus.queue import MessageBus
from agent.bus.event_bus import EventBus
from agent.core.events import AfterSendEvent
from agent.llm.client import LLMClient
from agent.lifecycle.turn import TurnLifecycle
from agent.tools.base import ToolRegistry
from agent.memory.store2 import MemoryStoreConfig
from agent.memory.service2 import MemoryServiceV2, create_memory_service_v2

logger = logging.getLogger(__name__)

# 工具执行轨迹：并发计数器 + 落盘开关（TOOL_TRACE=1 开启）
_tool_trace_active = 0

# 工具调用最大迭代次数（防止无限循环）
MAX_TOOL_ITERATIONS = 10

# 对话历史：最多保留的轮数（每轮=user+assistant 一对）
MAX_HISTORY_TURNS = 10
# 单轮估算最大 token（保守估计：4 chars ≈ 1 token）
MAX_HISTORY_CHARS = MAX_HISTORY_TURNS * 2000

# 行为准则（system prompt 的基础片段，可被 prompt_render 阶段扩展）
_BEHAVIOR_RULES = [
    "你是一个自然、真诚的助手。回答原则：",
    "1. 用户当前问什么就回答什么，不要把话题强行扯到用户之前提过的事情上",
    "2. 如果用户之前聊过跑步/咖啡，但现在在聊音乐，就专注聊音乐",
    "3. 不要主动把当前话题和用户的历史信息做关联，除非用户自己提起",
    "4. 保持对话自然流畅，像朋友聊天一样",
]


async def agent_loop(
    bus: MessageBus,
    llm: LLMClient,
    events: EventBus,
    tools: ToolRegistry | None = None,
    memory_service: MemoryServiceV2 | None = None,
) -> None:
    """Agent 主循环

    参数:
        bus:            消息总线（入站/出站队列）
        llm:            LLM 客户端
        events:         事件总线（作为 Phase 钩子链的扩展点）
        tools:          工具注册表（None 表示不使用工具）
        memory_service: 记忆服务 V2（None 表示不启用记忆）
    """

    lifecycle = TurnLifecycle(events)

    while True:
        # ── 1. 消费入站消息 ──────────────────────────────
        in_msg: InboundMessage = await bus.consume_inbound()
        logger.info(
            "收到消息: [%s] %s: %s",
            in_msg.channel, in_msg.sender_id, in_msg.content,
        )

        # ── 2. 加载相关记忆（在管线之前，供后续阶段注入） ──
        memory_block = ""
        if memory_service:
            try:
                memory_block = memory_service.build_memory_context()
                if memory_block:
                    logger.info("已加载相关记忆到上下文")
            except Exception as e:
                logger.warning("记忆加载失败（不影响主流程）: %s", e)

        # ── 3. before_turn 门 ────────────────────────────
        bt = await lifecycle.run_before_turn(in_msg, memory_block)
        if bt.abort:
            logger.info("消息被 before_turn 拦截: %s", bt.abort_reply or in_msg.content[:30])
            if bt.abort_reply:
                await _publish_abort(bus, in_msg, bt.abort_reply)
            continue

        # ── 4. before_reasoning 门 ───────────────────────
        br = await lifecycle.run_before_reasoning(bt)
        if br.abort:
            logger.info("消息被 before_reasoning 拦截: %s", br.abort_reply or in_msg.content[:30])
            if br.abort_reply:
                await _publish_abort(bus, in_msg, br.abort_reply)
            continue

        # ── 5. prompt_render 门（组装 messages） ──────────
        # 按 session_key 取对应的历史，避免跨通道串话
        history = memory_service.get_session_history_for_prompt(in_msg.session_key) if memory_service else []
        messages = await lifecycle.run_prompt_render(
            channel=in_msg.channel,
            chat_id=in_msg.chat_id,
            content=in_msg.content,
            system_parts=list(_BEHAVIOR_RULES),
            history=history,
            memory_block=br.memory_block,
        )
        # 调试：把发给 LLM 的完整 prompt 落盘（设 PROMPT_DUMP=1 开启）
        if os.environ.get("PROMPT_DUMP") == "1":
            _dump_prompt(messages, in_msg)

        # ── 6. 工具循环 ──────────────────────────────────
        tool_schemas = tools.to_openai_tools() if tools and tools.count > 0 else None
        reply_text = ""
        iterations = 0
        tools_used: list[str] = []

        try:
            while iterations < MAX_TOOL_ITERATIONS:
                # 6a. before_step 门（可提前终止）
                bs = await lifecycle.run_before_step(
                    channel=in_msg.channel,
                    chat_id=in_msg.chat_id,
                    iteration=iterations,
                    messages=messages,
                    tool_schemas=tool_schemas,
                )
                if bs.early_stop:
                    reply_text = bs.early_stop_reply or ""
                    break

                response = await llm.chat(
                    messages,
                    tools=tool_schemas,
                )

                if not response.has_tool_calls:
                    # LLM 返回文本回复 → 结束循环
                    reply_text = response.content or ""
                    messages.append({"role": "assistant", "content": reply_text})
                    break

                # ── 有工具调用 ────────────────────────────
                raw_calls = []
                for tc in response.tool_calls:
                    raw_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": raw_calls,
                })

                step_tools: list[str] = []

                # 并行执行所有工具调用（保持原顺序返回结果）
                async def _run_one(tc):
                    name = tc.function.name
                    args = _parse_arguments(tc.function.arguments)
                    logger.info("工具调用: %s %s", name, args)
                    _trace_tool_start(name)
                    try:
                        result = await _run_tool(tools, name, **args)
                    finally:
                        _trace_tool_end(name)
                    return name, result, tc.id

                tasks = [_run_one(tc) for tc in response.tool_calls]
                results = await asyncio.gather(*tasks)

                for name, result, call_id in results:
                    step_tools.append(name)
                    tools_used.append(name)
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call_id,
                    })

                iterations += 1
                # 6b. after_step 观察（快照）
                await lifecycle.run_after_step(
                    channel=in_msg.channel,
                    chat_id=in_msg.chat_id,
                    iteration=iterations,
                    tool_names=tuple(step_tools),
                    has_more=iterations < MAX_TOOL_ITERATIONS,
                )
            else:
                reply_text = (
                    f"我已经调用了 {MAX_TOOL_ITERATIONS} 次工具但仍未得出最终结果，"
                    "请简化你的需求。"
                )
                messages.append({"role": "assistant", "content": reply_text})

        except Exception:
            logger.exception("LLM/工具调用异常")
            reply_text = "抱歉，我在处理过程中遇到了一个内部错误。"

        # 兜底：空回复时生成一条占位消息
        if not reply_text or not reply_text.strip():
            reply_text = "（我没有生成回复内容，请重试）"

        # ── 7. after_reasoning 门（可改写最终回复） ───────
        ar = await lifecycle.run_after_reasoning(
            channel=in_msg.channel,
            chat_id=in_msg.chat_id,
            content=in_msg.content,
            reply=reply_text,
            tools_used=tuple(tools_used),
        )
        reply_text = ar.reply or reply_text

        # ── 8. 萃取记忆 ─────────────────────────────────
        if memory_service and reply_text:
            try:
                facts = await memory_service.extract_facts(
                    user_message=in_msg.content,
                    assistant_message=reply_text,
                )
                for fact in facts:
                    # 生成 source_ref 用于幂等去重
                    source_ref = f"{in_msg.session_key}_turn_{id(in_msg)}"
                    await memory_service.add_fact(fact, source_ref)
                if facts:
                    logger.info("萃取了 %d 条事实并存入记忆", len(facts))
            except Exception as e:
                logger.warning("记忆萃取失败（不影响主流程）: %s", e)

        # ── 9. 更新对话历史 ──────────────────────────────
        # 追加本轮 user + assistant（不包含中间工具调用）
        # 按 session_key 隔离，存入 MemoryStoreV2 的会话历史
        if memory_service:
            memory_service.append_session_history(in_msg.session_key, "user", in_msg.content)
            memory_service.append_session_history(in_msg.session_key, "assistant", reply_text)

        # ── 10. after_turn 观察（快照） ───────────────────
        await lifecycle.run_after_turn(
            channel=in_msg.channel,
            chat_id=in_msg.chat_id,
            content=in_msg.content,
            reply=reply_text,
            tools_used=tuple(tools_used),
        )

        # ── 11. 发布出站消息 ─────────────────────────────
        out_msg = OutboundMessage(
            channel=in_msg.channel,
            chat_id=in_msg.chat_id,
            content=reply_text,
            metadata={"reply_to_msg_id": id(in_msg)},
        )
        await bus.publish_outbound(out_msg)

        # ── 12. enqueue 后台并发通知 ─────────────────────
        events.enqueue(AfterSendEvent(
            channel=in_msg.channel,
            chat_id=in_msg.chat_id,
            content=reply_text,
        ))


# ── 内部辅助 ──────────────────────────────────────────────


def _trace_tool_start(name: str) -> None:
    """工具开始执行：记时间戳 + 并发数（TOOL_TRACE=1 时落盘）"""
    global _tool_trace_active
    _tool_trace_active += 1
    if os.environ.get("TOOL_TRACE") == "1":
        _append_tool_trace(f"START {name}  active={_tool_trace_active}")


def _trace_tool_end(name: str) -> None:
    """工具执行结束：记时间戳 + 并发数"""
    global _tool_trace_active
    _tool_trace_active -= 1
    if os.environ.get("TOOL_TRACE") == "1":
        _append_tool_trace(f"END   {name}  active={_tool_trace_active}")


def _append_tool_trace(line: str) -> None:
    """把工具轨迹追加写入 tool_trace.log（调试串行/并行用）"""
    try:
        import datetime

        with open("tool_trace.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now():%H:%M:%S.%f}] {line}\n")
    except Exception:
        logger.warning("工具轨迹落盘失败（不影响主流程）")


def _dump_prompt(messages: list[dict], in_msg: InboundMessage) -> None:
    """把发给 LLM 的完整 prompt 追加写入 prompt_dump.log（调试会话串话用）"""
    try:
        import datetime

        with open("prompt_dump.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{datetime.datetime.now():%H:%M:%S}] "
                    f"channel={in_msg.channel} chat={in_msg.chat_id} "
                    f"session={in_msg.session_key}\n")
            for m in messages:
                role = m.get("role", "?")
                content = m.get("content") or ""
                f.write(f"--- {role} ---\n{content}\n")
    except Exception:
        logger.warning("prompt 落盘失败（不影响主流程）")


async def _publish_abort(bus: MessageBus, in_msg: InboundMessage, content: str) -> None:
    """拦截时若提供了 abort_reply，则作为出站消息发出。"""
    await bus.publish_outbound(OutboundMessage(
        channel=in_msg.channel,
        chat_id=in_msg.chat_id,
        content=content,
    ))


def _parse_arguments(raw: str) -> dict:
    """解析工具参数字符串（API 返回的是 JSON 字符串）"""
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        logger.warning("工具参数解析失败: %s", raw)
        return {}


async def _run_tool(
    tools: ToolRegistry | None,
    name: str,
    **kwargs,
) -> str:
    """执行工具调用，捕获异常并返回结果字符串"""
    if tools is None:
        return f"错误: 工具系统未初始化"

    tool = tools.get(name)
    if tool is None:
        available = [t.name for t in tools.list_all()]
        return f"错误: 未知工具 '{name}'，可用工具: {available}"

    try:
        result = await tool.execute(**kwargs)
        return str(result)
    except Exception as e:
        logger.exception("工具执行失败: %s", name)
        return f"工具 '{name}' 执行出错: {e}"
