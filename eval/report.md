================================================================
Agent 边界评测报告
================================================================
用例: 4 个 | 断言: 11 条 | 通过: 11 | 失败: 0

[PASS] 多工具并行执行  (工具调用 · mock)
    LLM 一次返回多个独立工具调用时，应并行执行而非串行（慢工具并发探针验证）
    ✓ 工具 A 被调用
    ✓ 工具 B 被调用
    ✓ 工具 C 被调用
    ✓ 3 个独立工具应并行执行（最大并发 >= 3）
    信息: 出站消息 1 条 | LLM 调用 2 次
    prompt 日志: D:\agent_project\myAgent\eval\logs\parallel_tools.md

[PASS] 双通道会话隔离  (多轮上下文 · mock)
    Telegram 与 Console 两个通道应各自独立记忆历史，互相不串话
    ✓ Telegram 第3轮历史不应包含 Console 的 Python/SQL 内容
    ✓ Telegram 第3轮历史应包含自己会话的跑步内容
    信息: 出站消息 3 条 | LLM 调用 3 次
    prompt 日志: D:\agent_project\myAgent\eval\logs\session_isolation.md

[PASS] 工具异常处理  (鲁棒性 · mock)
    工具抛出异常时 agent 不应崩溃，应把错误信息回填给 LLM 继续迭代
    ✓ fail_tool 被调用
    ✓ 工具错误信息应回填给 LLM（prompt 含错误详情）
    ✓ 异常后应继续迭代而不是死循环
    信息: 出站消息 1 条 | LLM 调用 2 次
    prompt 日志: D:\agent_project\myAgent\eval\logs\tool_error.md

[PASS] 工具循环收敛  (工具调用 · mock)
    LLM 反复请求工具时应在上限内收敛，防止死循环（默认上限 MAX_TOOL_ITERATIONS=10）
    ✓ 工具循环应在上限 10 次内终止，不无限调用
    ✓ echo_tool 至少被调用一次
    信息: 出站消息 1 条 | LLM 调用 10 次
    prompt 日志: D:\agent_project\myAgent\eval\logs\tool_loop_bounded.md

----------------------------------------------------------------
失败项 = 当前边界/痛点，可作为改进方向与面试素材
================================================================