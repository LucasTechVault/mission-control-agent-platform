from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mission_control.inference.requests import ModelMessage

from mission_control.tools.contracts import ToolCall, ToolResult

class AgentStatus(StrEnum):
    """
    High-level lifecycle state of 1 Mission Control agent investigation run.
    """
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentState(BaseModel):
    """
    Canonical runtime state for 1 Mission Control agent investigation.
    
    This state belongs to Mission Control, not to the model.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True) # prevent hallucination and immutability validation
    
    # Investigation Identity
    run_id: str = Field(default_factory=lambda str: uuid4())
    objective: str = Field(min_length=1)
    
    # Model-facing context representation
    messages: list[ModelMessage] = Field(default_factory=list)
    
    # Execution history (Runtime tracks)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[ToolResult] = Field(default_factory=list)
    
    # Runtime Lifecycle handling
    step_count: int = Field(
        default=0,
        ge=0,
    )
    
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: AgentStatus = AgentStatus.RUNNING
    
    final_answer: str | None = None
    failures: str | None = None
    
    @classmethod
    def create(
        cls,
        *,
        objective: str,
        system_prompt: str
    ) -> "AgentState":
        """
        Create initial state for a new investigation.
        """
        return cls(
            objective=objective,
            message=[
                ModelMessage(
                    role="system",
                    content=system_prompt
                ),
                ModelMessage(
                    role="user",
                    content=objective
                )
            ]
        )