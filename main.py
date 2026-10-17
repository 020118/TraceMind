"""Console/Telegram entry point for TraceMind.

FastAPI uses the same :class:`agent.runtime.AgentRuntime` through
``uvicorn api.app:app``.  Keeping assembly in one runtime prevents channel
specific entry points from drifting apart.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agent.runtime import AgentRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def serve(
    config_path: str = "config.toml",
    workspace: str | None = None,
    *,
    enable_telegram: bool = True,
    enable_proactive: bool = True,
) -> None:
    """Start the shared runtime with interactive channels enabled."""

    runtime = AgentRuntime.from_config_path(
        config_path,
        workspace=workspace,
        enable_console=True,
        enable_telegram=enable_telegram,
        enable_proactive=enable_proactive,
    )
    await runtime.start()
    try:
        await asyncio.Event().wait()
    finally:
        await runtime.stop()


def _get_flag_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        raise ValueError(f"参数 {flag} 缺少值")
    return args[idx + 1]


def _has_flag(args: list[str], flag: str) -> bool:
    """Return whether a boolean CLI flag is present."""
    return flag in args


if __name__ == "__main__":
    cli_args = sys.argv[1:]
    selected_config = "config.toml"
    selected_workspace: str | None = None

    try:
        if value := _get_flag_value(cli_args, "--config"):
            selected_config = value
        if value := _get_flag_value(cli_args, "--workspace"):
            selected_workspace = value
        if _has_flag(cli_args, "--help") or _has_flag(cli_args, "-h"):
            print(
                "用法: python main.py [--config PATH] [--workspace PATH] "
                "[--no-telegram] [--no-proactive]"
            )
            print("  --no-telegram   仅运行 Console，不连接 Telegram Bot")
            print("  --no-proactive  不启动主动推送后台任务")
            raise SystemExit(0)
        asyncio.run(
            serve(
                selected_config,
                selected_workspace,
                enable_telegram=not _has_flag(cli_args, "--no-telegram"),
                enable_proactive=not _has_flag(cli_args, "--no-proactive"),
            )
        )
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
