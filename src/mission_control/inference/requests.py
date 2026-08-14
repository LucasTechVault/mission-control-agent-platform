from typing import Literal, Any
from uuid import uuid4

from pydantic import BaseModel, Field

MessageRole = Literal[
    "system",
    "user",
    "assistant"
]

class ModelMessage(BaseModel):
    """A single message provided to a chat model."""
    
    role: MessageRole
    content: str = Field(min_length=1)

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
    