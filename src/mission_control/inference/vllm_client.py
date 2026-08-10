import time

import httpx
import structlog
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger(__name__)

# '_' prefix specifies class for internal use only
# (The Letter)- handles the actual "meat" of the response
class _VLLMMessage(BaseModel):
    model_cfg = ConfigDict(extra="ignore") # extra fields are ignored, not forbidden
    
    content: str | None = None
    reasoning: str | None = None

# (The Envelope) - LLMs can generate multiple different answers to same prompt (choices)
# This class wraps the msg & adds a finish reason (success or failure)
class _VLLMChoice(BaseModel):
    model_cfg = ConfigDict(extra="ignore")
    
    msg: _VLLMMessage
    finish_reason: str | None = None

# (The Receipt) - Tracks token consumption
class _VLLMUsage(BaseModel):
    model_cfg = ConfigDict(extra="ignore")
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class _VLLMResponse(BaseModel):
    model_cfg = ConfigDict(extra="ignore")
    
    model: str
    choices: list[_VLLMChoice]
    usage: _VLLMUsage