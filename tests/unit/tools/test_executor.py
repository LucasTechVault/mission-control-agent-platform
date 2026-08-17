from pathlib import Path

import pytest

from mission_control.tools.contracts import (
    ToolCall,
    ToolExecutionStatus,
)
from mission_control.tools.executor import (
    ToolExecutor,
)
from mission_control.tools.investigation.service_health import (
    GetServiceHealthTool,
    JsonServiceHealthProvider,
)
from mission_control.tools.registry import (
    ToolRegistry,
)


@pytest.fixture
def executor() -> ToolExecutor:

    provider = (
        JsonServiceHealthProvider(
            Path(
                "configs/tools/service_health.json"
            )
        )
    )

    tool = GetServiceHealthTool(
        provider
    )

    registry = ToolRegistry()
    registry.register(tool)

    return ToolExecutor(
        registry
    )


@pytest.mark.asyncio
async def test_execute_registered_tool(
    executor: ToolExecutor,
) -> None:

    call = ToolCall(
        tool_name="get_service_health",
        arguments={
            "service": "onboarding-api",
            "environment": "prod",
        },
    )

    result = await executor.execute(
        call
    )

    assert (
        result.status
        == ToolExecutionStatus.SUCCESS
    )

    assert result.error is None

    assert (
        result.output["status"]
        == "degraded"
    )

    assert (
        result.output[
            "error_rate_percent"
        ]
        == 11.0
    )