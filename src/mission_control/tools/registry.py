from typing import Any

from mission_control.tools.base import Tool
from mission_control.tools.contracts import ToolDefinition

class ToolRegistryError(RuntimeError):
    """Base error for Mission Control tool registration."""

# Guard against human registration error
class DuplicateToolError(ToolRegistryError):
    """A tool name has been already been registered."""

# Guard against AI hallucinations
class ToolNotFoundError(ToolRegistryError):
    """Requested tool is not registered."""

class ToolRegistry:
    """
    Allowlisted collection of capabilities available to Mission Control.
    
    The model may only request tools present in this registry.
    """
    
    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}
        
    # The only way tools can be added to system
    def register(self, tool: Tool[Any, Any]) -> None:
        name = tool.definition.name
        
        if name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {name}")
    
        self._tools[name] = tool
    
    def get(self, name: str) -> Tool[Any, Any]:
        tool = self._tools.get(name)
        
        if tool is None:
            raise ToolNotFoundError(f"Tool is not registered: {name}") # guard from hallucinations
    
        return tool

    # Single, comprehensive menu of every capability available at time of invocation
    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]
    
    def names(self) -> list[str]:
        return list(self._tools.keys())
    