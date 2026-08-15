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

class ToolCall(BaseModel):
    """_summary_
    Normalized request for tool invocation.
    
    A ToolCall is a proposal for execution by the LLM.
    It is not proof that execution has been authorized.
    
    Note: Normalized means parsed into defined contract / schema.
    """
    
    model_config = ConfigDict(extra="forbid") # prevent hallucination
    
    call_id: str = Field(
        default_factory=lambda: str(uuid4())
    )
    
    tool_name: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    
    arguments: dict[str, Any] = Field(
        default_factory=dict
    )
    
    parent_request_id: str | None = None

class ToolError(BaseModel):
    """_summary_
    Normalized deterministic tool failure.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    code: str = Field(
        min_length=1
    )
    
    message: str = Field(
        min_length=1
    )
    
    retryable: bool = False

class ToolExecutionStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"

class ToolResult(BaseModel):
    """_summary_
    Normalized result returned by Mission Control's deterministic execution layer.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    call_id: str
    tool_name: str
    
    status: ToolExecutionStatus
    
    output: Any | None = None # Possible tool failure, thus no output
    error: ToolError | None = None # Normalize error, if any
    
    duration_ms: float | None = Field(
        default=None,
        ge=0.0, # Can't have negative execution time.
    )
    
    @model_validator(mode="after")
    def validate_result_state(self, ) -> "ToolResult":
        
        # Validate contradictory status and error
        if self.status == ToolExecutionStatus.SUCCESS and self.error is not None:
            raise ValueError("Successful tool result cannot contain an error.")
    
        if self.status == ToolExecutionStatus.ERROR and self.error is None:
            raise ValueError("Failed tool result must contain an error.")
        
        return self
    