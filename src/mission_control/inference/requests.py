from typing import Literal, Any, ConfigDict
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from mission_control.tools.contracts import (
    ToolDefinition,
)

from mission_control.tools.contracts import (
    ToolCall,
    ToolDefinition,
)

MessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool"
]

class ModelMessage(BaseModel):
    """
    Framework neutral message stored by Mission Control.
    
    Assistant message may contain model-proposed ToolCalls.
    
    Tool messages carry deterministic observations (ToolResult) produced by Mission Control.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    role: MessageRole
    
    content: str | None = None
    
    tool_calls: list[ToolCall] = Field(default_factory=list) # AI proposal for ToolCall(s)
    
    tool_call_id: str | None = None
    
    @model_validator(mode="after")
    def validate_message_shape(self) -> "ModelMessage": # Python forward reference
        has_content = bool(self.content and self.content.strip())
        
        # 1. System / User role message validation
        if self.role in {"system", "user"}:
            # 1.1 Empty content validation
            if not has_content:
                raise ValueError(f"{self.role} message requires content.")
            
            # 1.2 toolcall guard - system & user cannot call tools
            if self.tool_calls:
                raise ValueError(f"{self.role} message cannot contain tool calls.")
        
            if self.tool_call_id is not None:
                raise ValueError(f"{self.role} message cannot contain tool_call_id.")
        
        # 2. Assistant role message validation
        elif self.role == "assistant":
            
            # 2.1 - no message AND no tool call
            if not has_content and not self.tool_calls:
                raise ValueError("Assistant message requires either content or tool call proposal.")

            # 2.2 - tool_call_id protection - 1 Assistant can have MANY tool calls
            # IDs must reside within ToolCall obj, not in ModelMessage 
            if self.tool_call_id is not None:
                raise ValueError("Assistant message cannot contain tool_call_id.")
        
        # 3. Tool message validation
        elif self.role == "tool":
            if not has_content:
                raise ValueError("Tool message requires content.")
            
            # 3.1 tool_call uuid to map between ToolCall & ToolResult
            if self.tool_call_id is None:
                raise ValueError("Tool message requires tool_call_id.")

            # 3.2 recursive tool call - tool calling other tools
            if self.tool_calls:
                raise ValueError("Tool message cannot contain new tool calls.")
        
        return self # Returns the object that was being checked, else will be None

        
class ModelRequest(BaseModel):
    """Mission Control's internal contract for 1 model generation."""
    
    request_id: str = Field(
        default_factory=lambda: str(uuid4()) # Ensure fresh unique value generated for every instantiation
    )
    
    messages: list[ModelMessage] = Field(
        min_length=1
    )
    
    # controls randomness of model's output
    # lower = deterministic, higher = more "creative"
    temperature: float = Field( 
        default=0.2,
        ge=0.0,
        le=2.0,
    )
    
    # for nucleus sampling - another way to control output diversity
    top_p: float = Field(
        default=0.8, # consider smallest set of tokens whose cumulative prob. > 80%
        gt=0.0,
        le=1.0,
    )
    
    # absolute max len of generated response.
    max_tokens: int = Field(
        default=512,
        gt=0,
        le=8192,
    )
    
    enable_thinking: bool = Field(
        default=False
    )
    
    # Optional structured-output contract
    response_schema: dict[str, Any] | None = None
    response_schema_name: str | None = None
    
    # Timeout, None = use app default.
    timeout_seconds: float | None = Field(
        default=None,
        gt=0.0,
    )
    
    # Tool menu for LLM
    tool_definitions: list[ToolDefinition] = Field(default_factory=list)
    
    # Control Agent's autonomy - whether model allowed to use tool or not
    tool_choice: Literal[
        "none",
        "auto",
        "required"
    ] = "none"
    
    # Runs after Pydantic builds ModelRequest
    @model_validator(mode="after")
    def validate_tool_configuration(self) -> "ModelRequest": # Force model to use tool but did not provide any tool
        if self.tool_choice != "none" and not self.tool_definitions:
            raise ValueError("tool_choice requires at least one tool definition.")
        return self