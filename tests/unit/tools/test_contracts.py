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