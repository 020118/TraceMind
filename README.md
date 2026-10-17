# TraceMind · 可追溯的个人 AI Agent

一个从零实现的异步个人 Agent Runtime。`Trace` 代表可追溯的工具执行、消息投递与故障恢复，`Mind` 代表长期记忆、用户偏好与个人知识。项目覆盖工具调用、动态工具发现、文档 RAG、代码修改验证、主动推送、多通道消息和 Web 工作台。

它不只是一次 LLM 调用：模型可以发现并执行工具，把结果回填后继续推理；系统会保存会话与记忆，也能基于本地资料检索并返回可追溯引用。

---

## 功能概览

| 模块 | 当前能力 |
|---|---|
| Agent Runtime | OpenAI 兼容模型、Function Calling、多步工具循环、步数与超时保护、7 阶段生命周期扩展 |
| 工具系统 | `BaseTool + ToolRegistry`、风险/别名/来源元数据、`tool_search`、延迟 Schema 加载、并行与串行执行 |
| Coding Agent | workspace 内文件列表/读取/写入/精确编辑/受控删除、Python 语法检查、受限 pytest 验证 |
| 长期记忆 | Memory V2 分层文件、Pending 缓冲、定期合并、语义+关键词混合检索、会话历史持久化 |
| 文档 RAG | PDF/Markdown/文本/代码解析、Chunking、Embedding、Qdrant Dense + 关键词 + RRF、citation、故障降级 |
| 多通道 | Web / FastAPI / Console / Telegram，共用同一套 `AgentRuntime` 和消息总线；Web 支持独立多会话与最近使用排序 |
| 流式交互 | LLM 真正增量输出；工具参数碎片重组；NDJSON 推送模型、工具和错误事件 |
| 主动 Agent | Gate / Source / Judge / Resolve / Deliver，支持冷却、去重、证据和投递记录 |
| 可靠性 | 删除操作二次确认与 journal、会话事务写入、发送重试与死信、Qdrant 补偿同步 |
| 部署 | React/Vite + FastAPI 多阶段镜像，Docker Compose 编排 Agent 与 Qdrant，健康检查与持久卷 |

---

## 架构

```text
React Web / HTTP / Console / Telegram
                 │
                 ▼
       MessageBus + ApiChannel
                 │
                 ▼
 Lifecycle → AgentLoop → LLMProvider → OpenAI-compatible LLM
                 │                         │
                 │       tool_calls        │ content deltas
                 ▼                         ▼
 ToolRegistry / tool_search        NDJSON stream events
                 │                         │
   ┌─────────────┼─────────────┐           ▼
   │             │             │       React 增量渲染
文件/代码      Web/MCP       Knowledge tools
                               │
                    SQLite RAG + Qdrant
```

### 文档 RAG 数据流

```text
workspace 文档
  → 解析标题/行号/PDF 页码
  → Chunk + overlap
  → SHA-256 内容去重
  → Embedding
  → SQLite 保存原文、元数据、向量副本与同步状态
  → Qdrant collection 保存检索向量
  → Dense + keyword + RRF
  → Top-K 原文片段与 citation
```

Qdrant 不可用时自动退回 SQLite 精确 Dense；Embedding 也不可用时退回关键词检索。

---

## 技术栈

- 后端：Python 3.11+、asyncio、FastAPI、Uvicorn、Pydantic
- 模型：OpenAI 兼容 API（DeepSeek、通义千问或其他兼容服务）
- 前端：React 19、TypeScript、Vite
- 数据：SQLite、Qdrant、DashScope/OpenAI-compatible Embedding、NumPy
- 协议与通道：MCP（JSON-RPC 2.0 over stdio）、HTTP、Telegram
- 文档：pypdf；Markdown/文本/代码结构化解析
- 工程：Docker、Docker Compose、pytest、pytest-asyncio

---

## 快速开始

### 1. 配置

复制示例环境变量和配置文件，不要把真实密钥提交到 Git：

```powershell
Copy-Item .env.example .env
Copy-Item example_config.toml config.toml
```

至少填写：

```dotenv
AGENT_MAIN_API_KEY=你的模型密钥
AGENT_EMBEDDING_API_KEY=你的Embedding密钥
AGENT_API_KEY=可选的HTTP接口访问密钥
QDRANT_URL=http://qdrant:6333
```

`config.toml` 支持 `${ENV_NAME}` 占位符。Telegram、主动推送和高德地图均为可选能力。

### 2. Docker Compose（推荐演示方式）

```bash
docker compose up --build
```

启动后：

- Web 工作台：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`
- Agent 健康与就绪：`/health`、`/ready`
- Qdrant Dashboard：`http://127.0.0.1:6333/dashboard`

Compose 会启动两个容器：`tracemind` 和 `qdrant`。Agent 工作区与 Qdrant 数据分别使用持久卷；`knowledge_docs` 以只读方式挂载到 Agent workspace。

停止服务：

```bash
docker compose down
```

### 3. 本地后端

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn api.app:app --host 127.0.0.1 --port 8000 --env-file .env
```

若 Qdrant 在宿主机运行，把 `QDRANT_URL` 设置为 `http://localhost:6333`。只运行 CLI 可执行：

```powershell
python main.py
```

### 4. 前后端分离开发

后端保持在 `8000`，另开终端启动 Vite：

```powershell
Set-Location frontend
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。Vite 会把 `/v1`、`/health` 和 `/ready` 代理到 FastAPI，因此修改 React/CSS 后无需重建后端镜像。

---

## HTTP API

| 接口 | 作用 |
|---|---|
| `GET /health` | 进程存活检查 |
| `GET /ready` | Runtime、RAG 与向量库就绪状态 |
| `POST /v1/chat` | 非流式对话，兼容原调用方 |
| `POST /v1/chat/stream` | NDJSON 流式对话与工具执行事件 |
| `GET /v1/sessions` | 按最近使用时间列出 Web 会话 |
| `POST /v1/sessions` | 使用 UUID4 创建唯一空会话 |
| `GET /v1/sessions/{session_id}/messages` | 恢复会话消息 |
| `POST /v1/operations/{operation_id}/confirm` | 确认待执行的破坏性操作 |
| `POST /v1/operations/{operation_id}/cancel` | 取消待执行操作 |
| `GET /v1/rag/metrics` | Qdrant 与 SQLite shadow evaluation 指标 |

普通请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"session_id":"demo-user","message":"列出知识库中的文档"}'
```

如果未设置 `AGENT_API_KEY`，可省略 `X-API-Key`。同一个 `session_id` 会映射为稳定会话，不同用户应使用不同 ID。

Web 会话的短期消息、工具链和待确认操作按 `session_id` 隔离；个人偏好与长期记忆仍在同一个个人 Agent workspace 内共享。当前 API Key 对应一个个人用户边界，如果扩展为多用户 SaaS，需要再引入真实 `user_id` 并在所有会话查询中校验所有权。

FastAPI 模式默认不启动 Telegram 和主动调度器；需要时设置 `AGENT_ENABLE_TELEGRAM=1` 或 `AGENT_ENABLE_PROACTIVE=1`。

---

## RAG 使用方式

通过对话让 Agent 调用以下工具：

- `knowledge_index_file`：把 workspace 内文件加入知识库。
- `knowledge_search`：执行混合检索并返回 citation。
- `knowledge_list_documents`：列出已索引文档。
- `knowledge_retrieval_metrics`：查看 Recall@K、成功率、P50/P95 延迟和平均加速比。

默认 collection 为 `agent_knowledge_v1`。SQLite 数据位于 workspace 的 `knowledge/rag.db`，保存文档、chunks、Embedding 副本、Qdrant 同步状态和评测数据；Qdrant 专门负责近邻向量检索。

---

## 项目结构

```text
api/                         FastAPI、流式接口、静态前端托管
frontend/                    React + TypeScript + Vite 工作台
agent/
├── looping/core.py          当前 Agent Loop 与结构化流式事件
├── provider.py              模型 SDK 到内部响应类型的适配层
├── llm/client.py            OpenAI 兼容客户端与 delta/tool-call 重组
├── lifecycle/               7 阶段生命周期和拓扑排序
├── tools/                   工具注册、搜索、文件、代码、RAG、Web、MCP
├── rag/                     文档解析、SQLite store、Qdrant store、混合检索
├── memory/                  V1/V2 记忆与检索
├── reliability/             操作 journal 与可靠性状态
├── bus/                     消息总线与事件总线
└── channels/                API、Console、Telegram、Dispatcher
proactive_v2/                主动 Agent 管线
session/                     会话与消息事务存储
eval/                        离线评测与 Mock LLM
tests/                       单元、回归与集成测试
Dockerfile                   Node/Python 多阶段镜像
docker-compose.yml           Agent + Qdrant 编排
```

---

## 测试与验证

```powershell
python -m pytest -q
Set-Location frontend
npm run build
```

最近一次在依赖完整环境中的整仓结果为 `141 passed, 1 skipped`，前端 Vite 构建通过。测试覆盖工具注册与动态发现、文件边界与删除确认、语法/pytest 工具、可靠性恢复、记忆检索、RAG、Qdrant 集成、流式 LLM、ApiChannel 和 FastAPI。

测试大量使用 Fake LLM、Mock Embedding/HTTP/Qdrant，保证离线、可复现且不消耗模型费用。发布前仍应在新构建的 Docker 镜像中重跑一次完整测试。

---

## 已解决的典型问题

1. 多通道竞争同一个 `asyncio.Queue` 导致丢消息：改为单一 Dispatcher 按 channel 路由。
2. 工具数量增长导致 Schema Token 与选择干扰：增加工具元数据索引、`tool_search` 和延迟加载。
3. 代码修改无法证明正确：形成“读/改 → 语法检查 → pytest → 根据失败继续修复”闭环。
4. 删除等破坏性操作不应直接执行：使用唯一 `operation_id`、二次确认和持久化 journal。
5. Qdrant 是外部服务，可能不可用：保留 SQLite 精确 Dense 与关键词降级，并记录同步状态和 shadow metrics。
6. 前端动画不等于真实流式：模型 delta、工具事件和最终结果通过 NDJSON 端到端传输。

---

## 当前边界与后续计划

- [ ] 用真实研究资料构建 RAG 评测集，补 Recall@K、MRR/nDCG、引用正确率和答案忠实度
- [ ] 增加 Rerank、OCR、目录增量同步和知识库管理页面
- [ ] 在现有 Outbox 基础上增加持久化 Inbox、渠道 ACK 对账、统一 trace/span、客户端断开取消与断线续传
- [ ] GitHub Actions：测试、镜像构建、漏洞扫描与发布
- [ ] Nginx/Caddy、HTTPS、限流、密钥轮换与正式云部署
- [ ] 将不可信代码运行拆到短生命周、默认断网的独立 Runner 容器或 microVM

---

## 相关文档

- `PROACTIVE_MODULE_DESIGN.md`：主动 Agent 模块与边界设计
- `frontend/README.md`：前端本地开发与 Docker 运行方式
- `eval/README.md`：离线评测集、A/B 实验与可复现运行方式
