from pydantic import BaseModel, ConfigDict

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
    prompt_tokens: int # num tokens consumed by user
    completion_tokens: int # num tokens model generated in response
    total_tokens: int # prompt + completion tokens
    latency_ms: float # total round trip time taken (including network overhead)
    