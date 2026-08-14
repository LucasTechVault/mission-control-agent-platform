import pytest
from pydantic import ValidationError

from mission_control.investigation.models import (
    InvestigationBrief,
    Severity,
)


def test_investigation_brief_accepts_valid_contract() -> None:
    brief = InvestigationBrief(
        objective="Investigate onboarding failures",
        summary="Failures increased after Tuesday's deployment.",
        severity=Severity.HIGH,
        hypotheses=[
            "A deployment introduced a regression.",
        ],
        evidence_needed=[
            "Deployment history",
            "Failure-rate metrics",
        ],
        recommended_actions=[
            "Compare failures before and after the deployment.",
        ],
    )

    assert brief.severity is Severity.HIGH
    assert len(brief.hypotheses) == 1


def test_investigation_brief_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        InvestigationBrief(
            objective="Investigate onboarding failures",
            summary="Failure rate increased.",
            severity="terrible",
            hypotheses=["Deployment regression"],
            evidence_needed=["Logs"],
            recommended_actions=["Inspect logs"],
        )


def test_investigation_brief_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        InvestigationBrief(
            objective="Investigate onboarding failures",
            summary="Failure rate increased.",
            severity="high",
            hypotheses=["Deployment regression"],
            evidence_needed=["Logs"],
            recommended_actions=["Inspect logs"],
            hallucinated_field="should not exist",
        )