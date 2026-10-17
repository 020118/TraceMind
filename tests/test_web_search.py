""" WebSearch / WebFetch 工具测试

运行方式：
    pytest tests/test_web_search.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.web_search import _clean_html, _strip_html, _clean_text


# ── _clean_html 测试 ───────────────────────────────────

def test_clean_html_removes_tags():
    """去掉简单标签"""
    result = _clean_html("<b>歌词</b>")
    assert result == "歌词"


def test_clean_html_removes_nested():
    """去掉嵌套标签"""
    result = _clean_html("<div><p>丁世光 <strong>神探</strong></p></div>")
    assert result == "丁世光 神探"


def test_clean_html_empty():
    """空输入返回空字符串"""
    assert _clean_html("") == ""
    assert _clean_html("<br/>") == ""
    assert _clean_html("   ") == ""


def test_clean_html_no_html():
    """没有 HTML 标签时原文返回"""
    result = _clean_html("你好 世界")
    assert result == "你好 世界"


# ── _strip_html 测试（你来写） ─────────────────────────
def test_strip_html():
    """更复杂的HTML标签"""
    result = _strip_html("hello <script>alert('hi')</script>world")
    assert result == "hello world"

# _clean_text测试
def test_clean_text():
    result = _clean_text("    hello world    ")
    assert result == "hello world"

def test_clean_text_removes_empty_lines():
    """行为③：跳过空行"""
    result = _clean_text("第一行\n\n\n第二行")
    assert result == "第一行\n第二行"


def test_clean_text_dedups_consecutive():
    """行为④：去重连续重复行"""
    result = _clean_text("相同\n相同\n不同\n相同")
    assert result == "相同\n不同\n相同"   # 连续的两行"相同"被合并


def test_clean_text_keeps_non_consecutive():
    """行为④边界：不相邻的重复行保留"""
    result = _clean_text("A\nB\nA")
    assert result == "A\nB\nA"          # 两个 A 不相邻，都保留


def test_clean_text_empty():
    """空输入"""
    assert _clean_text("") == ""

