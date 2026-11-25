"""Security & safety guardrails for TraceMind Agent runtime.

Covers:
1. SSRF URL / DNS / IP validation for outbound HTTP fetches.
2. Sensitive path blocking for filesystem tools.
3. Secret / credential redaction (DLP) before LLM / user exposure.
4. Untrusted external tool output framing against indirect prompt injection.
5. Lightweight JSON Schema validation for tool arguments before execution.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ── 1. Secret / DLP Redaction ────────────────────────────────────────────────

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # PEM private keys
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # OpenAI / DeepSeek / DashScope / Anthropic style sk- tokens
    (
        re.compile(r"\bsk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_\-]{16,}\b"),
        "[REDACTED_API_KEY]",
    ),
    # GitHub personal access / OAuth tokens
    (
        re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    # AWS access key IDs
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED_AWS_KEY]",
    ),
    # Bearer tokens
    (
        re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9\-._~+/=]{20,}"),
        r"\1[REDACTED_TOKEN]",
    ),
    # Key-value credential assignments (e.g. api_key="...", password=...)
    (
        re.compile(
            r"""(?i)\b((?:api[_-]?key|secret[_-]?key|access[_-]?token|bot[_-]?token|password|passwd)\s*[:=]\s*["']?)([^\s"'`,;]{12,})(["']?)"""
        ),
        r"\1[REDACTED]\3",
    ),
    # URLs with embedded user:password@host
    (
        re.compile(r"(https?://[^/\s:@]+:)([^/\s@]+)(@)"),
        r"\1[REDACTED]\3",
    ),
)


def redact_secrets(text: str) -> str:
    """Mask likely API keys, tokens, and private keys in text."""
    if not text:
        return text
    result = text
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ── 2. SSRF Protection ───────────────────────────────────────────────────────

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
        "instance-data",
    }
)


def _is_forbidden_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_http_url(url: str) -> str:
    """Validate that a URL is a public http(s) target and not an SSRF vector.

    Resolves DNS and rejects loopback, RFC1918 private, link-local (including
    cloud metadata 169.254.169.254), multicast, and reserved IP ranges.
    """
    clean = str(url or "").strip()
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"}:
        raise PermissionError(f"安全拦截：仅允许 http/https 协议，拒绝 {parsed.scheme or '空协议'}")
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise PermissionError("安全拦截：URL 缺少有效主机名")
    if hostname in _BLOCKED_HOSTNAMES or hostname.endswith(".local") or hostname.endswith(".internal"):
        raise PermissionError(f"安全拦截：禁止访问本地或内网主机 {hostname}")

    # Direct IP literal check first
    try:
        if _is_forbidden_ip(hostname):
            raise PermissionError(f"安全拦截：禁止访问内网或保留地址 {hostname}")
        return clean
    except ValueError:
        pass

    # Resolve DNS and check every returned IP address
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise PermissionError(f"安全拦截：无法解析主机 {hostname}（{exc}）") from exc

    if not addr_infos:
        raise PermissionError(f"安全拦截：主机 {hostname} 无可用解析地址")

    for info in addr_infos:
        sockaddr = info[4]
        ip_str = str(sockaddr[0])
        if _is_forbidden_ip(ip_str):
            raise PermissionError(f"安全拦截：主机 {hostname} 解析到禁止访问的内网地址 {ip_str}")

    return clean


# ── 3. Sensitive Path Protection ─────────────────────────────────────────────

_ALLOWED_ENV_EXAMPLES = frozenset({".env.example", ".env.sample", ".env.template"})
_SENSITIVE_FILENAMES = frozenset({"config.toml", "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"})
_SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".kdbx"})


def check_sensitive_path(
    path: Path,
    allowed_dir: Path | None,
    *,
    for_write: bool = False,
) -> None:
    """Block reading or mutating sensitive credentials and runtime state files."""
    name = path.name.lower()
    suffix = path.suffix.lower()

    if (name == ".env" or name.startswith(".env.")) and name not in _ALLOWED_ENV_EXAMPLES:
        raise PermissionError(f"安全拦截：禁止访问环境变量密钥文件 {path.name}")
    if name in _SENSITIVE_FILENAMES or suffix in _SENSITIVE_SUFFIXES:
        raise PermissionError(f"安全拦截：禁止访问敏感配置或私钥文件 {path.name}")
    if name == "sessions.db" or name.startswith("sessions.db-"):
        raise PermissionError(f"安全拦截：禁止访问 Agent 会话数据库 {path.name}")

    if allowed_dir is not None:
        try:
            relative = path.resolve().relative_to(allowed_dir.resolve())
        except ValueError:
            return
        parts = [p.lower() for p in relative.parts]
        if not parts:
            return
        first = parts[0]
        if first in {".agent_state", ".trash"}:
            raise PermissionError(f"安全拦截：路径 {relative.as_posix()} 属于 Agent 内部保留目录")
        if first == ".git":
            if for_write or (len(parts) >= 2 and parts[1] == "config"):
                raise PermissionError(f"安全拦截：禁止访问或修改 Git 内部配置 {relative.as_posix()}")


# ── 4. Untrusted Tool Output Framing ─────────────────────────────────────────

_UNTRUSTED_TOOL_NAMES = frozenset(
    {
        "web_fetch",
        "web_search",
        "knowledge_search",
    }
)


def wrap_untrusted_tool_output(tool_name: str, text: str) -> str:
    """Wrap external / untrusted tool results with explicit data boundaries."""
    if not text:
        return text
    if tool_name in _UNTRUSTED_TOOL_NAMES or tool_name.startswith("mcp_"):
        return (
            f'<untrusted_tool_output source="{tool_name}">\n'
            f"{text}\n"
            f"</untrusted_tool_output>"
        )
    return text


# ── 5. Lightweight JSON Schema Argument Validation ───────────────────────────

def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def validate_tool_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> str | None:
    """Return a clear corrective error string if tool arguments violate schema."""
    if not isinstance(arguments, dict):
        return "错误: 工具参数必须是 JSON 对象。"
    if "_raw" in arguments and len(arguments) == 1:
        return (
            "错误: 工具调用参数不是合法的 JSON 对象，请检查引号、逗号和转义后重新调用。"
        )
    if not schema or not isinstance(schema, dict):
        return None

    required = schema.get("required") or []
    missing = [key for key in required if key not in arguments or arguments[key] is None]
    if missing:
        return f"错误: 缺少必填参数 {', '.join(missing)}，请补全后重试。"

    properties = schema.get("properties") or {}
    for key, value in arguments.items():
        if key.startswith("_") or value is None:
            continue
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        expected_type = prop.get("type")
        if isinstance(expected_type, str) and not _matches_json_type(value, expected_type):
            return (
                f"错误: 参数 '{key}' 类型不匹配，期望 {expected_type}，"
                f"实际为 {type(value).__name__}，请修正后重试。"
            )
    return None
