# tool_search + 延迟工具加载 A/B 评测报告

- 模式：`deterministic`
- 模型：`deterministic-plan-provider`
- 数据集：`D:\agent_project\myAgent\eval\datasets\tool_discovery_ab.json`
- 每个 case 重复：`1` 次
- 生成时间：`2026-10-03T03:55:02.653756+00:00`

> 确定性模式只验证评测器、Agent Loop 和延迟加载机制，
> 不代表真实大模型的工具选择能力。面试数据应使用 `--mode real` 结果。

## 汇总对比

| 指标 | A：全量 Schema | B：tool_search 延迟加载 | B - A |
|---|---:|---:|---:|
| 任务成功率 | 100.0% | 100.0% | +0.0pp |
| 工具选择召回率 | 100.0% | 100.0% | +0.0pp |
| 工具选择准确率 | 100.0% | 100.0% | +0.0pp |
| 所需 Schema 加载率 | 100.0% | 100.0% | +0.0pp |
| 动态搜索调用率 | 0.0% | 100.0% | +100.0pp |
| 安全违规率 | 0.0% | 0.0% | +0.0pp |
| 无效/重复工具调用率 | 0.0% | 0.0% | +0.0pp |
| 平均初始 Schema 数 | 9.00 | 1.00 | -8.00 |
| 平均初始 Schema 字节 | 2681.00 | 794.00 | -1887.00 |
| 平均累计 Schema 字节 | 6511.00 | 7939.86 | +1428.86 |
| 平均 LLM 调用数 | 2.43 | 3.43 | +1.00 |
| 平均工具调用数 | 1.43 | 2.43 | +1.00 |
| 平均耗时（ms） | 16.38 | 18.37 | +1.99 |

## 分用例结果

| 版本 | 轮次 | 用例 | 成功 | 实际工具链 | 初始 Schema | LLM 调用 | 耗时 ms |
|---|---:|---|:---:|---|---:|---:|---:|
| baseline | 1 | read_python_file | PASS | read_file | 9 | 2 | 16.12 |
| baseline | 1 | list_project_files | PASS | list_dir | 9 | 2 | 15.95 |
| baseline | 1 | syntax_only | PASS | check_python_syntax | 9 | 2 | 15.06 |
| baseline | 1 | web_research | PASS | web_search | 9 | 2 | 14.39 |
| baseline | 1 | fetch_web_page | PASS | web_fetch | 9 | 2 | 13.95 |
| baseline | 1 | weather_lookup | PASS | get_weather | 9 | 2 | 22.05 |
| baseline | 1 | fix_and_verify_code | PASS | read_file → edit_file → check_python_syntax → run_tests | 9 | 5 | 17.13 |
| deferred | 1 | read_python_file | PASS | tool_search → read_file | 1 | 3 | 15.61 |
| deferred | 1 | list_project_files | PASS | tool_search → list_dir | 1 | 3 | 17.19 |
| deferred | 1 | syntax_only | PASS | tool_search → check_python_syntax | 1 | 3 | 21.26 |
| deferred | 1 | web_research | PASS | tool_search → web_search | 1 | 3 | 22.82 |
| deferred | 1 | fetch_web_page | PASS | tool_search → web_fetch | 1 | 3 | 18.76 |
| deferred | 1 | weather_lookup | PASS | tool_search → get_weather | 1 | 3 | 16.76 |
| deferred | 1 | fix_and_verify_code | PASS | tool_search → read_file → edit_file → check_python_syntax → run_tests | 1 | 6 | 16.20 |

## 解读边界

- 本实验使用与生产工具同名、同类 Schema 的无副作用探针，隔离网络波动和真实文件修改。
- 评测了工具发现、选择和编排；工具实现正确性由 pytest 单元/集成测试保证。
- Schema 字节数是可重现的上下文体积代理指标，不是模型服务商的真实 Token 计费数。
- 真实模型具有随机性，正式数据建议每条 case 至少重复 3 次，并保留模型和数据集版本。
