from typing import Literal

import pytest
from pydantic import (
    Field,
    ValidationError,
)

from mission_control.tools.contracts import (
    ToolArguments,
    ToolCall,
    ToolDefinition,
    ToolError,
    ToolExecutionStatus,
    ToolResult,
    ToolSideEffect,
)

class GetServiceHealthArguments(
    ToolArguments
):
    service: str = Field(
        min_length=1,
        description=(
            "Service whose health should be checked."
        ),
    )

    environment: Literal[
        "dev",
        "uat",
        "prod",
    ] = Field(
        description=(
            "Deployment environment."
        ),
    )

from typing import Literal

import pytest
from pydantic import (
    Field,
    ValidationError,
)

from mission_control.tools.contracts import (
    ToolArguments,
    ToolCall,
    ToolDefinition,
    ToolError,
    ToolExecutionStatus,
    ToolResult,
    ToolSideEffect,
)
from mission_control.tools.schema import (
    build_tool_definition,
    validate_tool_arguments,
)

def test_build_tool_definition() -> None:

    definition = build_tool_definition(
        name="get_service_health",
        description=(
            "Retrieve the current health "
            "of an application service."
        ),
        arguments_model=(
            GetServiceHealthArguments
        ),
        side_effect=ToolSideEffect.READ,
    )

    assert isinstance(
        definition,
        ToolDefinition,
    )

    assert (
        definition.name
        == "get_service_health"
    )

    assert (
        definition.side_effect
        == ToolSideEffect.READ
    )

    assert (
        definition.input_schema["type"]
        == "object"
    )

    assert set(
        definition.input_schema["required"]
    ) == {
        "service",
        "environment",
    }

def test_validate_valid_tool_arguments() -> None:

    call = ToolCall(
        tool_name="get_service_health",
        arguments={
            "service": "onboarding-api",
            "environment": "prod",
        },
    )

    arguments = validate_tool_arguments(
        arguments_model=(
            GetServiceHealthArguments
        ),
        raw_arguments=call.arguments,
    )

    assert (
        arguments.service
        == "onboarding-api"
    )

    assert (
        arguments.environment
        == "prod"
    )

# Test invalid env
def test_reject_invalid_environment() -> None:

    call = ToolCall(
        tool_name="get_service_health",
        arguments={
            "service": "onboarding-api",
            "environment": "mars",
        },
    )

    with pytest.raises(
        ValidationError
    ):
        validate_tool_arguments(
            arguments_model=(
                GetServiceHealthArguments
            ),
            raw_arguments=call.arguments,
        )

# Test Hallucinated arguments
def test_reject_unknown_tool_argument() -> None:

    call = ToolCall(
        tool_name="get_service_health",
        arguments={
            "service": "onboarding-api",
            "environment": "prod",
            "force_restart": True,
        },
    )

    with pytest.raises(
        ValidationError
    ):
        validate_tool_arguments(
            arguments_model=(
                GetServiceHealthArguments
            ),
            raw_arguments=call.arguments,
        )

# Result invariant test
def test_successful_tool_result() -> None:

    result = ToolResult(
        call_id="call-123",
        tool_name="get_service_health",
        status=(
            ToolExecutionStatus.SUCCESS
        ),
        output={
            "status": "healthy"
        },
        duration_ms=12.5,
    )

    assert result.error is None

def test_failed_tool_result_requires_error() -> None:

    with pytest.raises(
        ValidationError
    ):
        ToolResult(
            call_id="call-123",
            tool_name="get_service_health",
            status=(
                ToolExecutionStatus.ERROR
            ),
        )

def test_error_tool_result() -> None:

    result = ToolResult(
        call_id="call-123",
        tool_name="get_service_health",
        status=(
            ToolExecutionStatus.ERROR
        ),
        error=ToolError(
            code="service_unavailable",
            message=(
                "Monitoring backend "
                "could not be reached."
            ),
            retryable=True,
        ),
    )

    assert result.error is not None
    assert result.error.retryable is True