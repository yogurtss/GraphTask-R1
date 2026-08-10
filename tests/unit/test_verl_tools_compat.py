import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _V05BaseTool:
    def __init__(self, config: dict[str, Any], tool_schema: Any) -> None:
        self.config = config
        self.name = tool_schema.function.name

    async def create(self, instance_id: str | None = None, **kwargs: Any) -> str:
        del kwargs
        return instance_id or "v05-session"


def test_verl_v05_plain_string_tool_api_is_supported(monkeypatch: Any) -> None:
    verl = types.ModuleType("verl")
    tools = types.ModuleType("verl.tools")
    base_tool = types.ModuleType("verl.tools.base_tool")
    schemas = types.ModuleType("verl.tools.schemas")
    base_tool.BaseTool = _V05BaseTool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.tools", tools)
    monkeypatch.setitem(sys.modules, "verl.tools.base_tool", base_tool)
    monkeypatch.setitem(sys.modules, "verl.tools.schemas", schemas)

    source = Path(__file__).parents[2] / "graphtask_r1/training/verl_tools.py"
    spec = importlib.util.spec_from_file_location("_test_verl_v05_tools", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class SolverTool(module._SessionTool):
        allowed_roles = frozenset({"solver"})

    schema = SimpleNamespace(function=SimpleNamespace(name="solver_tool"))
    tool = SolverTool({}, schema)
    created = asyncio.run(tool.create(role="solver", graph_snapshot="toy-v1"))
    assert created == "v05-session"
    assert module._tool_response("result") == "result"
    assert tool._sessions["v05-session"]["role"] == "solver"
