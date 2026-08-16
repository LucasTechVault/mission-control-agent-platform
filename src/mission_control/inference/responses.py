from pydantic import BaseModel, ConfigDict, Field

from mission_control.tools.contracts import (
    ToolCall,
)

class ModelResponse(BaseModel):
    """
    Mission Control's normalized model response
    normalized = convert raw into standard, consistent format.
    Defines exact data backend will receive after inference server is done processing.
    """
    
    # In Pydantic v2, if pass dict with extra fields not defined in class, Pydantic will just ignore
    model_config = ConfigDict(extra="forbid") # prevents dict with extra fields to be forbidden.
    
    request_id: str # uuid
    model: str # Exact model version that served request ("Qwen/Qwen3.6-27B")
    text: str | None # Nullable final generated output 
    reasoning: str | None # Nullable modern LLM's Chain-of-Thought
    finish_reason: str | None # Nullable explanation on why generation stopped
    
    tool_calls: list[ToolCall] = Field(default_factory=list) # if no tool called, default is empty list
     
    prompt_tokens: int # num tokens consumed by user
    completion_tokens: int # num tokens model generated in response
    total_tokens: int # prompt + completion tokens
    
    latency_ms: float # total round trip time taken (including network overhead)
    
class ModelStreamEvent(BaseModel):
    """_summary_
    One normalized event from a streaming model generation.
    
    Individual events may contain content, reasoning, completion metadata,
    or final usage information...
    """
    
    # These fields are ChatLM contracts & shouldn't be customized.
    model_config = ConfigDict(extra="forbid")
    
    request_id: str
    model: str
    
    text_delta: str | None = None
    reasoning_delta: str | None = None
    
    finish_reason: str | None = None
    
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    
    elapsed_ms: float
    time_to_first_token_ms: float | None = None
    
    done: bool = False