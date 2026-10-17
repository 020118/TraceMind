"""内置技能目录（迁移自 akashic agent.skills，仅保留路径常量）"""

from __future__ import annotations

from pathlib import Path

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
