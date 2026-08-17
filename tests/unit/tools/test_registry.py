from pathlib import Path

import pytest

from mission_control.tools.investigation.service_health import (
    GetServiceHealthTool,
    JsonServiceHealthProvider,
)
from mission_control.tools.registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
)


def build_tool() -> GetServiceHealthTool:

    provider = JsonServiceHealthProvider(
        Path(
            "configs/tools/service_health.json"
        )
    )

    return GetServiceHealthTool(
        provider
    )


def test_registry_returns_registered_tool() -> None:

    registry = ToolRegistry()
    tool = build_tool()

    registry.register(tool)

    resolved = registry.get(
        "get_service_health"
    )

    assert resolved is tool


def test_registry_rejects_unknown_tool() -> None:

    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError
    ):
        registry.get(
            "delete_everything"
        )


def test_registry_rejects_duplicate_tool() -> None:

    registry = ToolRegistry()

    registry.register(
        build_tool()
    )

    with pytest.raises(
        DuplicateToolError
    ):
        registry.register(
            build_tool()
        )