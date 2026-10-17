# Agent 边界评测

量化 agent 的边界与痛点，供改进与面试素材使用。

> 注意：原有 `eval.runner` 主要为旧 `agent.core.agent_loop` 保留的兼容回归。
> 新评测应优先直接运行生产使用的 `agent.looping.core.AgentLoop`。

## tool_search + 延迟加载 A/B 实验

第一组对比实验比较：

- A（baseline）：旧策略，每轮一次性暴露全部工具 Schema。
- B（deferred）：新策略，初始只暴露 `tool_search`，搜索后加载命中工具。

先用确定性模式验证评测器（零 API 成本）：

```powershell
python -m eval.tool_discovery_ab
```

再用 `config.toml` 的真实模型运行 A/B：

```powershell
python -m eval.tool_discovery_ab --mode real --runs 3
```

可以只跑一个用例或一个版本：

```powershell
python -m eval.tool_discovery_ab --mode real --runs 1 --case code
python -m eval.tool_discovery_ab --mode real --runs 1 --variant deferred
```

报告保存到：

- `eval/results/tool_discovery_ab_<mode>.md`
- `eval/results/tool_discovery_ab_<mode>.json`

报告包含任务成功率、工具选择准确率/召回率、工具搜索调用率、
Schema 加载率、无效工具调用率、初始/累计 Schema 字节、LLM 调用数和耗时。
确定性模式不代表真实模型能力；面试时应使用 `--mode real`、多次重复的结果。

## 只在本地 Console 试用代码工具

CLI 默认会启动 Telegram 和主动循环。如果只想测试文件读写和代码验证，使用：

```powershell
python main.py --no-telegram --no-proactive
```

这样只会启动 Console 和 Agent Loop，不会调用 Telegram `getMe` 接口。

## 运行

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m eval.runner                     # mock 模式（零成本、确定性回归）
python -m eval.runner --real              # 真实 LLM（读 config.toml [llm.main]）
python -m eval.runner --real --case 会话   # 真实 LLM 只跑指定用例
python -m eval.runner --list              # 列出用例
```

## 两种模式怎么选

| 模式 | 命令 | 用途 |
|------|------|------|
| mock | `python -m eval.runner` | 回归测试。脚本回放 LLM，测工具循环、异常、并行等确定性边界，便宜、可复现 |
| real | `python -m eval.runner --real` | 亲身体验。真实模型推理，适合看"prompt 到底混入了什么"（如会话串话） |

## 输出

- `eval/report.md` — 评测报告（PASS/FAIL + 失败原因）
- `eval/logs/<case>.md` — 每个 step 发给 LLM 的完整 prompt 日志（人工核对 agent 看到了什么）

## 用例说明

| 用例 | 测什么 | 预期 |
|------|--------|------|
| 双通道会话隔离 | telegram/console 历史是否串话 | PASS（按 session 隔离） |
| 多工具并行执行 | LLM 一次多工具调用是否并行 | PASS（`asyncio.gather`） |
| 工具异常处理 | 工具抛异常是否回填给 LLM | PASS |
| 工具循环收敛 | 是否在 10 次上限内终止 | PASS |

## 新增用例

在 `eval/cases/` 加一个 JSON：

```json
{
  "name": "用例名",
  "category": "分类",
  "description": "测什么",
  "steps": [
    {"channel": "telegram", "chat_id": "tg1", "sender_id": "u1", "content": "用户消息"}
  ],
  "script": [
    [{"type": "text", "content": "LLM 脚本回复"}]
  ],
  "assertions": [
    {"type": "llm_prompt_contains", "step": 0, "substrings": ["期望内容"]},
    {"type": "tool_called", "name": "工具名", "args": {"参数": "值"}},
    {"type": "llm_prompt_not_contains", "step": 0, "substrings": ["不应出现的内容"]}
  ]
}
```

断言类型见 `runner.py` 的 `_check_assertion`。
