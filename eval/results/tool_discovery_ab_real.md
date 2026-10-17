# tool_search + 延迟工具加载 A/B 评测报告

- 模式：`real`
- 模型：`deepseek-v4-flash`
- 数据集：`D:\agent_project\myAgent\eval\datasets\tool_discovery_ab.json`
- 每个 case 重复：`3` 次
- 生成时间：`2026-10-03T04:02:48.990983+00:00`

## 汇总对比

| 指标 | A：全量 Schema | B：tool_search 延迟加载 | B - A |
|---|---:|---:|---:|
| 基础设施错误率 | 0.0% | 0.0% | +0.0pp |
| 任务成功率 | 100.0% | 100.0% | +0.0pp |
| 工具选择召回率 | 100.0% | 100.0% | +0.0pp |
| 工具选择准确率 | 81.1% | 79.0% | -2.1pp |
| 所需 Schema 加载率 | 100.0% | 100.0% | +0.0pp |
| 动态搜索调用率 | 0.0% | 100.0% | +100.0pp |
| 安全违规率 | 0.0% | 0.0% | +0.0pp |
| 无效/重复工具调用率 | 41.2% | 42.3% | +1.1pp |
| 平均初始 Schema 数 | 9.00 | 1.00 | -8.00 |
| 平均初始 Schema 字节 | 2681.00 | 794.00 | -1887.00 |
| 平均累计 Schema 字节 | 7404.67 | 8349.00 | +944.33 |
| 平均 LLM 调用数 | 2.76 | 3.86 | +1.10 |
| 平均工具调用数 | 2.43 | 3.81 | +1.38 |
| 平均耗时（ms） | 3450.21 | 4194.09 | +743.88 |

## 分用例结果

| 版本 | 轮次 | 用例 | 成功 | 实际工具链 | 初始 Schema | LLM 调用 | 耗时 ms |
|---|---:|---|:---:|---|---:|---:|---:|
| baseline | 1 | read_python_file | PASS | read_file | 9 | 2 | 2028.10 |
| baseline | 1 | list_project_files | PASS | list_dir | 9 | 2 | 2263.17 |
| baseline | 1 | syntax_only | PASS | check_python_syntax | 9 | 2 | 1437.76 |
| baseline | 1 | web_research | PASS | web_search → web_search → web_search → web_fetch | 9 | 3 | 6524.07 |
| baseline | 1 | fetch_web_page | PASS | web_fetch | 9 | 2 | 3503.77 |
| baseline | 1 | weather_lookup | PASS | get_weather | 9 | 2 | 1533.34 |
| baseline | 1 | fix_and_verify_code | PASS | read_file → list_dir → read_file → read_file → edit_file → check_python_syntax → run_tests | 9 | 6 | 5538.45 |
| baseline | 2 | read_python_file | PASS | read_file | 9 | 2 | 2427.88 |
| baseline | 2 | list_project_files | PASS | list_dir | 9 | 2 | 2802.04 |
| baseline | 2 | syntax_only | PASS | check_python_syntax → list_dir | 9 | 2 | 2121.89 |
| baseline | 2 | web_research | PASS | web_search → web_search → web_fetch → web_search → web_search → web_search | 9 | 4 | 8139.88 |
| baseline | 2 | fetch_web_page | PASS | web_fetch → web_search | 9 | 3 | 4730.47 |
| baseline | 2 | weather_lookup | PASS | get_weather | 9 | 2 | 1244.36 |
| baseline | 2 | fix_and_verify_code | PASS | read_file → read_file → edit_file → check_python_syntax → run_tests | 9 | 5 | 4130.50 |
| baseline | 3 | read_python_file | PASS | read_file | 9 | 2 | 2173.46 |
| baseline | 3 | list_project_files | PASS | list_dir | 9 | 2 | 2257.75 |
| baseline | 3 | syntax_only | PASS | check_python_syntax | 9 | 2 | 1528.82 |
| baseline | 3 | web_research | PASS | web_search → web_search → web_search → web_search → web_fetch → web_search | 9 | 4 | 8473.47 |
| baseline | 3 | fetch_web_page | PASS | web_fetch | 9 | 2 | 3064.51 |
| baseline | 3 | weather_lookup | PASS | get_weather | 9 | 2 | 1660.94 |
| baseline | 3 | fix_and_verify_code | PASS | read_file → list_dir → read_file → edit_file → check_python_syntax → run_tests | 9 | 5 | 4869.72 |
| deferred | 1 | read_python_file | PASS | tool_search → tool_search → read_file | 1 | 3 | 3123.90 |
| deferred | 1 | list_project_files | PASS | tool_search → list_dir | 1 | 3 | 2839.90 |
| deferred | 1 | syntax_only | PASS | tool_search → tool_search → check_python_syntax | 1 | 3 | 2469.62 |
| deferred | 1 | web_research | PASS | tool_search → web_search → web_search → web_fetch → web_search | 1 | 5 | 7023.63 |
| deferred | 1 | fetch_web_page | PASS | tool_search → tool_search → web_fetch → web_search | 1 | 4 | 5205.54 |
| deferred | 1 | weather_lookup | PASS | tool_search → get_weather | 1 | 3 | 1907.63 |
| deferred | 1 | fix_and_verify_code | PASS | tool_search → list_dir → read_file → read_file → read_file → edit_file → check_python_syntax → run_tests | 1 | 7 | 6930.52 |
| deferred | 2 | read_python_file | PASS | tool_search → read_file | 1 | 3 | 2382.30 |
| deferred | 2 | list_project_files | PASS | tool_search → list_dir | 1 | 3 | 2395.49 |
| deferred | 2 | syntax_only | PASS | tool_search → tool_search → check_python_syntax → list_dir | 1 | 3 | 2633.65 |
| deferred | 2 | web_research | PASS | tool_search → web_search → web_search → web_fetch → web_search → web_fetch | 1 | 5 | 7224.00 |
| deferred | 2 | fetch_web_page | PASS | tool_search → tool_search → web_fetch | 1 | 3 | 4625.21 |
| deferred | 2 | weather_lookup | PASS | tool_search → get_weather | 1 | 3 | 2655.70 |
| deferred | 2 | fix_and_verify_code | PASS | tool_search → tool_search → read_file → list_dir → read_file → read_file → edit_file → check_python_syntax → run_tests | 1 | 6 | 5812.99 |
| deferred | 3 | read_python_file | PASS | tool_search → read_file → list_dir | 1 | 3 | 3648.99 |
| deferred | 3 | list_project_files | PASS | tool_search → list_dir | 1 | 3 | 2306.56 |
| deferred | 3 | syntax_only | PASS | tool_search → tool_search → check_python_syntax | 1 | 3 | 2821.07 |
| deferred | 3 | web_research | PASS | tool_search → web_search → web_search → web_search → web_fetch → web_search → web_search | 1 | 6 | 10017.90 |
| deferred | 3 | fetch_web_page | PASS | tool_search → web_fetch | 1 | 3 | 3743.63 |
| deferred | 3 | weather_lookup | PASS | tool_search → get_weather | 1 | 3 | 2325.92 |
| deferred | 3 | fix_and_verify_code | PASS | tool_search → read_file → read_file → edit_file → check_python_syntax → run_tests | 1 | 6 | 5981.70 |

## 解读边界

- 本实验使用与生产工具同名、同类 Schema 的无副作用探针，隔离网络波动和真实文件修改。
- 评测了工具发现、选择和编排；工具实现正确性由 pytest 单元/集成测试保证。
- Schema 字节数是可重现的上下文体积代理指标，不是模型服务商的真实 Token 计费数。
- 真实模型具有随机性，正式数据建议每条 case 至少重复 3 次，并保留模型和数据集版本。
