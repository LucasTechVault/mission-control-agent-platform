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