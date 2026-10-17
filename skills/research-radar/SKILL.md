---
name: research-radar
description: 根据用户的长期研究兴趣，轮换检索、去重和评估新资料，只在内容足够有价值时主动推送并保存依据。
---

# 个人研究雷达

## 目标

在每次 Drift 空闲轮次中只处理一个研究主题：从长期记忆确认用户兴趣，搜索少量新资料，完成去重和价值判断，并在确有增量时最多推送一条消息。

这个 skill 负责“按已有兴趣持续跟进”，不负责替用户猜测研究方向，也不在没有明确兴趣时频繁询问用户。

## 单次闭环

### 1. 读取连续性状态

先检查 `select_skill` 返回的 `local_context`：

- `cursor`：读取 `topic_index`、`last_checked_at`、`recent_seen_keys` 和 `recent_pushed_keys`。
- `journal_recent`：核对最近处理、拒绝和推送过的资料。
- `scratchpad`：仅在上轮为 `paused` 或 `waiting` 时用于恢复现场。

如果旧 cursor 缺少字段，按空值处理，不要报错。`recent_seen_keys` 和 `recent_pushed_keys` 都是有界列表，每次写回最多保留最新 100 项。

### 2. 确认研究兴趣

调用 `recall_memory`，查询“用户明确表达过的研究方向、正在做的项目、技术偏好、不感兴趣内容和主动推送偏好”。如果当前工具列表提供 `search_messages`，需要时再用它查找用户最近关于“研究方向”“关注”“想了解”“不感兴趣”的原话；没有该工具时不要尝试调用。

只采用用户明确表达过、且当前仍可合理视为有效的兴趣；不要根据职业、学校或闲聊擅自推断敏感偏好。

- 如果没有明确主题：以 `waiting/silent` 结束，在 `scratchpad_update` 说明“等待用户在对话中明确研究方向”，本轮不要主动追问。
- 如果用户明确要求不要主动推送：仍可静默更新检索状态，但不得调用 `message_push`。

### 3. 每轮只选一个主题

把确认到的主题按稳定顺序排列，用 `topic_index` 轮换选择一个主题；本轮只搜索这一个主题。下一轮把索引移动到下一个主题，避免热门主题长期挤占其他兴趣。

检索词应包含主题的关键实体，并优先寻找：

1. 官方文档、官方博客或项目公告；
2. 论文原文、预印本或作者主页；
3. 官方代码仓库和正式发布说明；
4. 可信技术媒体的原创报道。

先调用 `web_search` 获取候选；只对最相关的 1 至 2 项调用 `web_fetch` 核验正文。每轮最多评估 5 个候选，不做无边界搜索。

### 4. 规范化并去重

为每个候选生成稳定的 `content_key`：

- 有 DOI 时使用小写 DOI；
- arXiv 论文使用不含版本号的 arXiv ID；
- GitHub 项目使用小写的 `owner/repo`，发布版本可追加 tag；
- 其他网页使用规范化 URL：主机名小写、删除 fragment，并去掉 `utm_*`、`spm`、`ref` 等跟踪参数；
- 无可靠 URL 时，使用“规范化标题 + 来源域名”。

候选满足任一条件即视为重复，不再推送：

- `content_key` 已出现在 `recent_seen_keys` 或 `recent_pushed_keys`；
- 最近 journal 中已有相同 key；
- 标题、核心结论和来源与已处理资料实质相同，只是转载或换链接。

即使不推送，也把本轮核验过的候选 key 写入 `recent_seen_keys`，避免下轮反复花费检索和模型调用。

### 5. 评分并决定是否打扰用户

对每个非重复候选记录以下分数，总分 10 分：

- 相关性 `relevance`：0 至 3；是否直接命中用户当前研究主题。
- 新颖性 `novelty`：0 至 2；相对已知内容是否有真实增量。
- 可信度 `credibility`：0 至 2；是否有可核验的一手来源和证据。
- 时效性 `timeliness`：0 至 2；是否近期发布或当前正需要。
- 可行动性 `actionability`：0 至 1；能否影响学习、实验或项目决策。

只有同时满足下列条件才允许主动推送：

- 总分至少 7；
- `relevance >= 2` 且 `credibility >= 1`；
- 已通过原始页面核验标题、来源和关键结论；
- 不违反用户的负面偏好和主动推送偏好；
- 与近期已推送内容不重复。

信息不足、来源打不开或关键事实无法核验时不要补全猜测；应静默结束，或在工具暂时故障时保存为 `paused`。

### 6. 推送格式

满足阈值时，只选择本轮最高分的一项调用一次 `message_push`。消息保持简短，并包含：

```text
【研究雷达】标题
新发现：一句话说明新增信息。
为什么与你相关：指出对应的研究方向或当前任务。
依据：来源类型、发布时间和关键证据。
链接：原始来源 URL
局限：存在不确定性时用一句话说明。
```

“为什么与你相关”必须来自已检索到的用户兴趣证据，不要写泛化推荐语。

## 状态延续

每轮都通过 `finish_drift` 更新统一状态，不创建 `state.json` 或其他并行历史文件。

`cursor_update` 至少写入：

```json
{
  "topic_index": 1,
  "last_topic": "agent reliability",
  "last_checked_at": "ISO-8601 时间",
  "recent_seen_keys": ["候选 key"],
  "recent_pushed_keys": ["已推送 key"]
}
```

- `topic_index` 指向下一轮要处理的主题。
- `recent_seen_keys` 合并旧值和本轮已核验值，去重后只保留最新 100 项。
- 只有 `message_push` 成功后才能把 key 加入 `recent_pushed_keys`。
- 不在 cursor 或 journal 中保存整篇文章、密钥、账号信息或不必要的个人信息。

`journal_append` 为每个完成决策的候选追加一条记录：

```json
{
  "entry_type": "candidate_evaluated",
  "key": "content_key",
  "payload": {
    "topic": "主题",
    "title": "标题",
    "url": "原始来源",
    "score": 8,
    "decision": "pushed 或 rejected",
    "reason": "推送或拒绝的简短依据"
  }
}
```

## 异常与收尾

- 已完成检索且无高价值内容：`finish_drift(status="completed", message_result="silent", ...)`。
- 推送成功：`finish_drift(status="completed", message_result="sent", ...)`，并记录推送依据。
- 搜索或页面访问出现暂时性故障，且本轮无法形成可靠判断：`finish_drift(status="paused", message_result="silent", scratchpad_update="故障和下轮恢复点", ...)`。
- 没有明确研究兴趣，或正在等待用户改变推送偏好：`finish_drift(status="waiting", message_result="silent", scratchpad_update="等待的条件", ...)`。
- `message_push` 失败时不得把候选标记为已推送；保存为 `paused/silent`，供下轮重新判断。

无论是否推送，都必须调用一次 `finish_drift` 结束本轮。
