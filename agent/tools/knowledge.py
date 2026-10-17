"""Agent tools for the document knowledge base."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from agent.rag.service import RAGService
from agent.tools.base import BaseTool, ToolRegistry


class KnowledgeIndexFileTool(BaseTool):
    name = "knowledge_index_file"
    description = (
        "把 workspace 内的文档加入 RAG 知识库。适用于研究资料、Markdown、"
        "文本、代码、配置或 PDF；不要用它保存用户偏好，用户偏好属于个人记忆。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "workspace 内的文件路径"},
            "title": {"type": "string", "description": "可选的文档标题"},
        },
        "required": ["path"],
    }

    def __init__(self, service: RAGService) -> None:
        self._service = service

    async def execute(self, path: str, title: str = "", **_: Any) -> str:
        result = await self._service.ingest_file(path, title=title)
        state = "内容未变化，沿用现有索引" if result.unchanged else "索引已更新"
        mode = "向量+关键词" if result.semantic_indexed else "关键词（向量服务未配置或不可用）"
        return (
            f"{state}：{result.source_path}，共 {result.chunk_count} 个分块，"
            f"检索模式：{mode}，向量库状态：{result.vector_status}。"
        )


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = (
        "检索已经加入本地 RAG 知识库的研究文档，返回相关原文片段和可追溯来源。"
        "用于基于资料回答问题；不是网络搜索，也不是查询个人偏好记忆。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "需要从资料中查找的问题"},
            "top_k": {
                "type": "integer", "description": "返回片段数，默认 6，最大 20",
                "minimum": 1, "maximum": 20,
            },
        },
        "required": ["query"],
    }

    def __init__(self, service: RAGService) -> None:
        self._service = service

    async def execute(self, query: str, top_k: int = 6, **_: Any) -> str:
        hits = await self._service.search(query, top_k=top_k)
        if not hits:
            return "知识库中没有找到相关内容。请先用 knowledge_index_file 建立索引。"
        payload = {
            "query": query,
            "results": [
                {
                    "text": hit.chunk.text,
                    "heading": hit.chunk.heading,
                    "citation": hit.citation,
                    "source_path": hit.document.source_path,
                    "score": hit.score,
                    "semantic_score": hit.semantic_score,
                    "keyword_score": hit.keyword_score,
                }
                for hit in hits
            ],
            "instruction": "回答时请引用 citation，并区分资料原文与自己的推断。",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class KnowledgeListDocumentsTool(BaseTool):
    name = "knowledge_list_documents"
    description = "列出当前 RAG 知识库已经建立索引的文档及分块数量。"

    def __init__(self, service: RAGService) -> None:
        self._service = service

    async def execute(self, **_: Any) -> str:
        documents = self._service.list_documents()
        if not documents:
            return "知识库为空。"
        return json.dumps(
            [asdict(document) for document in documents],
            ensure_ascii=False,
            indent=2,
        )


class KnowledgeRetrievalMetricsTool(BaseTool):
    name = "knowledge_retrieval_metrics"
    description = (
        "查看 RAG 检索量化指标，对比 SQLite 精确 Dense 扫描与 Qdrant 检索的"
        "Recall@K、成功率、P50/P95 延迟和平均加速比。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "统计最近多少次检索，默认500，最大5000",
                "minimum": 1,
                "maximum": 5000,
            }
        },
        "required": [],
    }

    def __init__(self, service: RAGService) -> None:
        self._service = service

    async def execute(self, limit: int = 500, **_: Any) -> str:
        return json.dumps(
            self._service.retrieval_metrics(limit),
            ensure_ascii=False,
            indent=2,
        )


def register_knowledge_tools(registry: ToolRegistry, service: RAGService) -> None:
    registry.register(
        KnowledgeIndexFileTool(service),
        risk="write",
        aliases=("导入文档", "建立知识库", "文档索引", "RAG摄取"),
    )
    registry.register(
        KnowledgeSearchTool(service),
        risk="read-only",
        aliases=("知识库检索", "文档问答", "查询资料", "RAG检索"),
    )
    registry.register(
        KnowledgeListDocumentsTool(service),
        risk="read-only",
        aliases=("知识库列表", "已导入文档", "资料列表"),
    )
    registry.register(
        KnowledgeRetrievalMetricsTool(service),
        risk="read-only",
        aliases=("RAG评测", "向量检索对比", "Qdrant效果", "检索指标"),
    )


__all__ = [
    "KnowledgeIndexFileTool",
    "KnowledgeListDocumentsTool",
    "KnowledgeRetrievalMetricsTool",
    "KnowledgeSearchTool",
    "register_knowledge_tools",
]
