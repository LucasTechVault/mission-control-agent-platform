import os

import pytest

from mission_control.config import Settings
from mission_control.inference.vllm_client import VLLMModelGateway
from mission_control.investigation.brief_generator import (
    InvestigationBriefGenerator,
)
from mission_control.investigation.models import Severity


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_qwen_generates_structured_investigation_brief() -> None:
    if os.getenv(
        "MISSION_CONTROL_RUN_LIVE_INTEGRATION"
    ) != "1":
        pytest.skip(
            "Live inference test disabled."
        )

    settings = Settings()

    gateway = VLLMModelGateway(
        settings=settings,
    )

    generator = InvestigationBriefGenerator(
        gateway=gateway,
    )

    try:
        brief = await generator.generate(
            objective=(
                "Investigate the increase in customer "
                "onboarding failures."
            ),
            context=(
                "Customer onboarding failures increased "
                "from 4% to 11% after Tuesday's deployment. "
                "Most failures now occur during identity "
                "verification. No root cause has yet been "
                "confirmed."
            ),
        )

    finally:
        await gateway.aclose()

    print()
    print("=== Investigation Brief ===")
    print(
        brief.model_dump_json(
            indent=2
        )
    )

    assert brief.objective
    assert brief.summary

    assert brief.severity in {
        Severity.LOW,
        Severity.MEDIUM,
        Severity.HIGH,
        Severity.CRITICAL,
    }

    assert brief.hypotheses
    assert brief.evidence_needed
    assert brief.recommended_actions