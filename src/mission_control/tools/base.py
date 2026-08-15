from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from mission_control.tools.contracts import (
    ToolArguments,
    ToolDefinition,
)

# Placeholder for any class inheriting ToolArguments
ArgumentsT = TypeVar("ArgumentsT", bound=ToolArguments)

# Placeholder for any class inheriting BaseModel, specifically for Output from Tools
# Every tool returns different shape. New tools can be invented. Output just needs to be structured Pydantic obj
OutputT = TypeVar ("OutputT", bound=BaseModel)

class Tool(Protocol, Generic[ArgumentsT, OutputT]):
    """_summary_
    Framework-neutral Mission Control tool contract.
    
    A tool exposes:
    - model-facing definition
    - argument model
    - deterministic execution
    """
    
    @property
    def definition(self) -> ToolDefinition:
        ...
    
    @property
    def arguments_model(self) -> type[ArgumentsT]:
        ...
    
    async def execute(self, arguments: ArgumentsT,) -> OutputT:
        ...
    