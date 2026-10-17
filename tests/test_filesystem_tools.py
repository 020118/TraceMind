from __future__ import annotations

import pytest

from agent.tools.base import ToolRegistry
from agent.tools.filesystem import register_filesystem_tools


@pytest.mark.asyncio
async def test_filesystem_tools_are_registered_and_scoped(tmp_path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    expected = {"read_file", "list_dir", "edit_file", "write_file", "delete_file"}
    assert expected.issubset({tool.name for tool in registry.list_all()})

    await registry.get("write_file").execute(
        path="sample.py",
        content="value = 1\n",
    )
    read_result = await registry.get("read_file").execute(path="sample.py")
    assert "value = 1" in read_result

    edit_result = await registry.get("edit_file").execute(
        path="sample.py",
        old_text="value = 1",
        new_text="value = 2",
    )
    assert "已成功编辑" in edit_result
    assert "-value = 1" in edit_result
    assert "+value = 2" in edit_result


@pytest.mark.asyncio
async def test_delete_file_requires_confirmation_and_moves_to_trash(tmp_path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    await registry.get("write_file").execute(path="remove_me.txt", content="keep me")
    tool = registry.get("delete_file")

    blocked = await tool.execute(path="remove_me.txt")
    assert "必须经过用户明确确认" in blocked
    assert (tmp_path / "remove_me.txt").exists()

    deleted = await tool.execute(
        path="remove_me.txt",
        _confirmed=True,
        _operation_id="test-delete-operation",
    )
    assert "已删除文件" in deleted
    assert not (tmp_path / "remove_me.txt").exists()
    assert list((tmp_path / ".trash").rglob("remove_me.txt"))

    repeated = await tool.execute(
        path="remove_me.txt",
        _confirmed=True,
        _operation_id="test-delete-operation",
    )
    assert "之前已经成功" in repeated


@pytest.mark.asyncio
async def test_filesystem_tools_reject_paths_outside_workspace(tmp_path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    result = await registry.get("read_file").execute(path="..\\outside.txt")
    assert "超出允许目录" in result


@pytest.mark.asyncio
async def test_edit_file_requires_unambiguous_match(tmp_path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    await registry.get("write_file").execute(
        path="duplicate.py",
        content="value = 1\nvalue = 1\n",
    )
    result = await registry.get("edit_file").execute(
        path="duplicate.py",
        old_text="value = 1",
        new_text="value = 2",
    )
    assert "出现了 2 次" in result


@pytest.mark.asyncio
async def test_atomic_write_preserves_original_when_replace_fails(
    tmp_path, monkeypatch
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)
    target = tmp_path / "important.txt"
    target.write_text("original", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr("agent.host_tools.filesystem.os.replace", fail_replace)
    result = await registry.get("write_file").execute(
        path="important.txt",
        content="new content",
    )

    assert "写入文件失败" in result
    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".important.txt.*.tmp"))


@pytest.mark.asyncio
async def test_mutation_tools_reject_agent_internal_state(tmp_path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    result = await registry.get("write_file").execute(
        path=".agent_state/reliability.db",
        content="corrupt",
    )

    assert "内部保留目录" in result
    assert not (tmp_path / ".agent_state" / "reliability.db").exists()
