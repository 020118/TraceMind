# TraceMind RAG quality smoke benchmark

离线、确定性的 RAG 检索质量对比；不调用外部 LLM 或 Embedding API。

- 文档数：3
- 查询数：4
- Top-K：3
- Embedding：`offline-hash-embedding-64`

| 指标 | Baseline（RRF） | Improved（RRF + MMR） | 变化 |
|---|---:|---:|---:|
| recall_at_k | 1.0 | 1.0 | 0.0 |
| mrr | 1.0 | 1.0 | 0.0 |
| ndcg_at_k | 1.0 | 1.0 | 0.0 |
| citation_coverage | 1.0 | 1.0 | 0.0 |
| empty_result_rate | 0.0 | 0.0 | 0.0 |
| distinct_sources_at_k | 1.5 | 1.75 | 0.25 |
| average_latency_ms | 0.99275 | 0.9725 | -0.02025 |

## Query Details

| Query | Baseline sources | Improved sources | Improved nDCG@K |
|---|---|---|---:|
| runtime_tools | docs/agent_runtime.md | docs/agent_runtime.md | 1.0 |
| personal_memory | docs/personal_memory.md, docs/rag.md | docs/personal_memory.md, docs/rag.md | 1.0 |
| document_rag | docs/rag.md | docs/rag.md | 1.0 |
| memory_and_rag | docs/rag.md, docs/personal_memory.md | docs/rag.md, docs/personal_memory.md, docs/agent_runtime.md | 1.0 |
