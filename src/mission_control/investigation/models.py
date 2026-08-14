from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

class Severity(StrEnum):
    """Supported Mission Control investigation severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InvestigationBrief(BaseModel):
    """_summary_
    Structured result produced from initial investigation request.
    
    This is Mission Control domain contract, not a vLLM response model.
    """
    model_config = ConfigDict(
        extra="forbid",
    )
    
    objective: str = Field(
        min_length=1,
        description="The investigation objective being assessed.",
    )
    
    summary: str = Field(
        min_length=1,
        description="Concise summary of current situation.",
    )
    
    severity: Severity = Field(
        min_length=1,
        description="Estimated severity based only on supplied information."
    )
    
    hypotheses: list[str] = Field(
        min_length=1,
        description="Plausible explanations that should be investigated."
    )
    
    evidence_needed: list[str] = Field(
        min_length=1,
        description="Evidence required to confirm or reject hypotheses."
    )
    
    recommended_actions: list[str] = Field(
        min_length=1,
        description="Immediate Investigation actions, not destructive remediation."
    )