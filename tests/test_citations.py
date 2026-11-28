from __future__ import annotations

import json

from api.citations import extract_citations


def test_extract_citations_from_knowledge_and_web_traces():
    kb_payload = json.dumps(
        {
            "query": "TraceMind",
            "results": [
                {
                    "citation": "architecture.md#L10-L25",
                    "source_path": "docs/architecture.md",
                    "heading": "Agent Loop",
                    "text": "AgentLoop uses an event-driven architecture.",
                    "score": 0.912,
                },
                {
                    "citation": "architecture.md#L10-L25",
                    "source_path": "docs/architecture.md",
                    "heading": "Agent Loop",
                    "text": "Duplicate entry should be deduplicated.",
                    "score": 0.85,
                },
            ],
        },
        ensure_ascii=False,
    )
    web_result = (
        '<untrusted_tool_output source="web_search">\n'
        "搜索结果: FastAPI\n"
        "========================================\n\n"
        "1. FastAPI Documentation\n"
        "   链接: https://fastapi.tiangolo.com/tutorial/\n"
        "   摘要: Modern, fast web framework for building APIs.\n\n"
        "2. Starlette\n"
        "   链接: https://www.starlette.io/\n"
        "   摘要: ASGI toolkit.\n"
        "</untrusted_tool_output>"
    )
    trace = [
        {"name": "knowledge_search", "arguments": {"query": "TraceMind"}, "result": kb_payload},
        {"name": "web_search", "arguments": {"query": "FastAPI"}, "result": web_result},
        {
            "name": "web_fetch",
            "arguments": {"url": "https://www.starlette.io/"},
            "result": "Already seen URL should deduplicate",
        },
        {
            "name": "web_fetch",
            "arguments": {"url": "https://docs.python.org/3/"},
            "result": "Python 3 documentation overview",
        },
        {
            "name": "web_fetch",
            "arguments": {"url": "http://127.0.0.1/admin"},
            "result": "抓取失败: 安全拦截",
        },
    ]

    citations = extract_citations(trace)
    assert len(citations) == 4
    assert citations[0]["kind"] == "knowledge"
    assert citations[0]["title"] == "architecture.md#L10-L25"
    assert citations[0]["score"] == 0.912

    urls = [c["url"] for c in citations if c["kind"] == "web"]
    assert urls == [
        "https://fastapi.tiangolo.com/tutorial/",
        "https://www.starlette.io/",
        "https://docs.python.org/3/",
    ]
