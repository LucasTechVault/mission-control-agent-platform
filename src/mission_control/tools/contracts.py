from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

class ToolArguments(BaseModel):
    """_summary_
    Base class for every Mission Control tool's argument.
    
    Unknown arguments are rejected rather than silently ignored.
    """
    
    model_config = ConfigDict(extra="forbid") # prevents hallucinated argument keys

class ToolSideEffect(StrEnum):
    """_summary_
    High-level classification of what executing a tool may change.
    
    e.g.
    get_service_health() -> READ
    restart_service() -> WRITE
    """
    NONE = "none"
    READ = "read"
    WRITE = "write"

class ToolDefinition(BaseModel):
    """_summary_
    Model-facing description of 1 capability exposed by Mission Control.
    
    This is framework-neutral. It is NOT raw vLLM / OpenAI wire representation.
    """
    
    model_config = ConfigDict(extra="forbid") # prevent hallucination of keys
    
    name: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Stable identifier used when requesting the tool."
    )
    
    description: str = Field(
        min_length=1,
        description="Description used by the model to determine when the tool should be called."
    )
    
    input_schema: dict[str, Any] = Field(
        description="JSON Schema describing the tool arguments."
    )
    
    side_effect: ToolSideEffect = ToolSideEffect.NONE
    
    strict: bool = True
    