# TraceMind Dashboard

当前工作台包含：

- Agent 对话与历史消息恢复
- 服务端 UUID 会话创建、最近使用排序和独立会话切换；空白会话不会重复新建
- 真正的模型流式文本，而不是前端假打字动画
- 模型阶段、工具名称、输入、结果、耗时和状态过程（显示在对应回答下方）
- 删除文件的独立确认/取消弹窗
- API Key 会话级保存（只使用 `sessionStorage`）

流式链路使用 `POST /v1/chat/stream` 和 NDJSON。`src/api.ts` 通过
`fetch + ReadableStream + TextDecoder` 逐行解析事件，`src/App.tsx` 将
`answer_delta` 追加到同一条回答，并处理 `answer_reset`、工具事件、错误和最终结果。

页面展示的是可观察的 Agent 执行事件和工具证据，不是模型隐藏的私有思维链。

会话历史、工具过程和删除确认按 `session_id` 隔离；长期记忆仍属于当前个人 Agent workspace，因此不同会话可以共享稳定的个人偏好，但不会共享整段聊天记录。

## 本地开发

先在项目根目录启动 FastAPI：

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000 --env-file .env
```

再启动前端：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`。Vite 会把 `/v1`、`/health` 和 `/ready`
代理到 `http://127.0.0.1:8000`。

## Docker

根目录的多阶段 `Dockerfile` 会先构建前端，再把 `frontend/dist` 复制到
FastAPI 镜像。执行 `docker compose up --build` 后，直接打开
`http://127.0.0.1:8000`。

如果 `.env` 配置了 `AGENT_API_KEY`，在页面右上角输入相同值即可；前端
不会把这个值写入源码或 `localStorage`。
