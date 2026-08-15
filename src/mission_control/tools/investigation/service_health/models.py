from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator,

from mission_control.tools.contracts import ToolArguments

Environment = Literal["dev", "uat", "prod"]

class ServiceHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

# The object that will replace ArgumentsT placeholder (Input Contract)
# What the model will provide
class GetServiceHealthArguments(ToolArguments):
    """_summary_
    Arguments accepted by get_service_health tool.
    """
    
    service: str = Field(
        min_length=1,
        pattern=r"[a-zA-Z0-9_.-]+$",
        description="Canonical application service name."
    )
    
    environment: Environment = Field(
        description="Deployment environment to inspect."
    )

# Output contract - what the external provider must return
class ServiceHealthSnapshot(BaseModel):
    """_summary_
    Normalized health information returned by get_service_health tool.
    """
    
    model_config = ConfigDict(
        extra="forbid"
    )
    
    service: str = Field(
        min_length=1,
    )
    
    environment: Environment
    
    status: ServiceHealthStatus
    
    error_rate_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0 # error rate cannot be more than 100%
    )
    
    p95_latency_ms: float | None = Field(
        default=None,
        ge=0.0 # latency cannot be negative
    )
    
    active_incidents: int = Field(
        default=0,
        ge=0,
    )
    
    observed_at: datetime
    
    source: str = Field(
        min_length=1,
    )
    
    @field_validator("observed_at")
    @classmethod
    def observed_at_must_be_timezone_aware( # In distributed systems, time without timezone info is meaningless.
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include timezone information.")
    
        return value
    