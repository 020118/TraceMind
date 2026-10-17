"""Document parsing and structure-aware chunking."""

from __future__ import annotations

import re
from pathlib import Path

from agent.rag.models import ParsedSection

TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".rst", ".py", ".js", ".jsx", ".ts",
    ".tsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".toml",
    ".yaml", ".yml", ".json", ".csv",
}


def parse_document(path: Path) -> list[ParsedSection]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix not in TEXT_SUFFIXES:
        raise ValueError(
            f"暂不支持 {suffix or '无扩展名'} 文件；支持 Markdown、文本、代码、配置和 PDF"
        )
    return _parse_text(path.read_text(encoding="utf-8", errors="replace"))


def chunk_sections(
    sections: list[ParsedSection], *, chunk_size: int, chunk_overlap: int
) -> list[ParsedSection]:
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 2))
    chunks: list[ParsedSection] = []
    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            target_end = min(start + chunk_size, len(text))
            end = target_end
            if target_end < len(text):
                window = text[start:target_end]
                boundaries = [window.rfind(mark) for mark in ("\n\n", "。", ". ", "\n")]
                best = max(boundaries)
                if best >= chunk_size // 2:
                    end = start + best + (2 if window[best:best + 2] == "\n\n" else 1)
            piece = text[start:end].strip()
            if piece:
                line_offset = text[:start].count("\n")
                piece_lines = max(1, piece.count("\n") + 1)
                chunks.append(
                    ParsedSection(
                        text=piece,
                        heading=section.heading,
                        start_line=(section.start_line + line_offset) if section.start_line else None,
                        end_line=(section.start_line + line_offset + piece_lines - 1)
                        if section.start_line else None,
                        page_number=section.page_number,
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - chunk_overlap)
    return chunks


def _parse_text(content: str) -> list[ParsedSection]:
    lines = content.splitlines()
    sections: list[ParsedSection] = []
    heading = ""
    buffer: list[str] = []
    buffer_start = 1

    def flush(end_line: int) -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                ParsedSection(
                    text=text,
                    heading=heading,
                    start_line=buffer_start,
                    end_line=end_line,
                )
            )
        buffer = []

    for line_number, line in enumerate(lines, start=1):
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            flush(line_number - 1)
            heading = match.group(1).strip()
            buffer_start = line_number + 1
            continue
        if not buffer and not line.strip():
            buffer_start = line_number + 1
            continue
        buffer.append(line)
    flush(len(lines))
    return sections


def _parse_pdf(path: Path) -> list[ParsedSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要安装 pypdf：pip install pypdf") from exc
    reader = PdfReader(str(path))
    sections: list[ParsedSection] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(ParsedSection(text=text, page_number=index))
    return sections

