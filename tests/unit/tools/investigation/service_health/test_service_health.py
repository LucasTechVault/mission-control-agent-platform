from pathlib import Path

import pytest
from pydantic import ValidationError

from mission_control.tools.contracts import (
    ToolSideEffect,
)
from mission_control.tools.investigation.service_health import (
    GetServiceHealthArguments,
    GetServiceHealthTool,
    JsonServiceHealthProvider,
    ServiceHealthRecordNotFoundError,
    ServiceHealthStatus,
)


FIXTURE_PATH = Path(
    "configs/tools/service_health.json"
)


@pytest.fixture
def provider() -> JsonServiceHealthProvider:
    return JsonServiceHealthProvider(
        FIXTURE_PATH
    )


@pytest.fixture
def tool(
    provider: JsonServiceHealthProvider,
) -> GetServiceHealthTool:
    return GetServiceHealthTool(
        provider=provider
    )


def test_tool_definition_is_read_only(
    tool: GetServiceHealthTool,
) -> None:

    definition = tool.definition

    assert (
        definition.name
        == "get_service_health"
    )

    assert (
        definition.side_effect
        == ToolSideEffect.READ
    )

    assert definition.strict is True


def test_tool_schema_rejects_extra_fields(
    tool: GetServiceHealthTool,
) -> None:

    schema = (
        tool.definition.input_schema
    )

    assert (
        schema[
            "additionalProperties"
        ]
        is False
    )


def test_arguments_reject_invalid_environment(
) -> None:

    with pytest.raises(
        ValidationError
    ):
        GetServiceHealthArguments(
            service="onboarding-api",
            environment="mars",
        )


def test_arguments_reject_unknown_field(
) -> None:

    with pytest.raises(
        ValidationError
    ):
        GetServiceHealthArguments(
            service="onboarding-api",
            environment="prod",
            restart=True,
        )


@pytest.mark.asyncio
async def test_tool_returns_service_health(
    tool: GetServiceHealthTool,
) -> None:

    arguments = (
        GetServiceHealthArguments(
            service="onboarding-api",
            environment="prod",
        )
    )

    result = await tool.execute(
        arguments
    )

    assert (
        result.service
        == "onboarding-api"
    )

    assert (
        result.environment
        == "prod"
    )

    assert (
        result.status
        == ServiceHealthStatus.DEGRADED
    )

    assert (
        result.error_rate_percent
        == 11.0
    )

    assert (
        result.active_incidents
        == 1
    )


@pytest.mark.asyncio
async def test_unknown_service_is_rejected(
    tool: GetServiceHealthTool,
) -> None:

    arguments = (
        GetServiceHealthArguments(
            service="does-not-exist",
            environment="prod",
        )
    )

    with pytest.raises(
        ServiceHealthRecordNotFoundError
    ):
        await tool.execute(
            arguments
        )


@pytest.mark.asyncio
async def test_returned_snapshot_does_not_mutate_provider(
    provider: JsonServiceHealthProvider,
) -> None:

    first = await (
        provider.get_service_health(
            service="onboarding-api",
            environment="prod",
        )
    )

    first.active_incidents = 999

    second = await (
        provider.get_service_health(
            service="onboarding-api",
            environment="prod",
        )
    )

    assert (
        second.active_incidents
        == 1
    )