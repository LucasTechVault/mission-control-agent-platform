import json
from collections.abc import AsyncIterator

import pytest

from mission_control.agent.loop import (
    ManualAgentLoop,
)
from mission_control.inference.requests import (
    ModelRequest,
)
from mission_control.inference.responses import (
    ModelResponse,
    ModelStreamEvent,
)
from mission_control.tools.contracts import (
    ToolCall,
)
from mission_control.tools.executor import (
    ToolExecutor,
)
from mission_control.tools.runtime import (
    build_default_tool_registry,
)


class ScriptedGateway:
    """
    Fake deterministic model:

    Turn 1:
        requests get_service_health

    Turn 2:
        verifies it received the observation,
        then returns final answer.
    """

    def __init__(self) -> None:
        self.requests: list[
            ModelRequest
        ] = []

    async def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        self.requests.append(
            request
        )

        # ==================================
        # TURN 1 — ACT
        # ==================================

        if len(self.requests) == 1:

            return ModelResponse(
                request_id=request.request_id,
                model="scripted-model",
                text=None,
                reasoning=None,
                finish_reason="tool_calls",

                tool_calls=[
                    ToolCall(
                        call_id="call-001",
                        tool_name=(
                            "get_service_health"
                        ),
                        arguments={
                            "service":
                                "onboarding-api",
                            "environment":
                                "prod",
                        },
                        parent_request_id=(
                            request.request_id
                        ),
                    )
                ],

                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=1.0,
            )

        # ==================================
        # TURN 2 — OBSERVE
        # ==================================

        tool_messages = [
            message
            for message
            in request.messages
            if message.role == "tool"
        ]

        assert len(
            tool_messages
        ) == 1

        tool_message = (
            tool_messages[0]
        )

        assert (
            tool_message.tool_call_id
            == "call-001"
        )

        observation = json.loads(
            tool_message.content
            or "{}"
        )

        assert (
            observation["status"]
            == "success"
        )

        assert (
            observation["output"]
            ["status"]
            == "degraded"
        )

        # ==================================
        # TURN 2 — FINAL ANSWER
        # ==================================

        return ModelResponse(
            request_id=request.request_id,
            model="scripted-model",
            text=(
                "onboarding-api is currently "
                "degraded."
            ),
            reasoning=None,
            finish_reason="stop",
            tool_calls=[],
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
            latency_ms=1.0,
        )

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[
        ModelStreamEvent
    ]:

        # ManualAgentLoop intentionally uses
        # non-streaming generation in M03-I02.
        raise AssertionError(
            "stream() should not be called."
        )

        # Makes this function an async generator.
        if False:
            yield ModelStreamEvent(
                request_id=request.request_id,
                model="unused",
                elapsed_ms=0.0,
            )


@pytest.mark.asyncio
async def test_manual_agent_loop_observes_tool_result_and_finishes(
) -> None:

    registry = (
        build_default_tool_registry()
    )

    executor = ToolExecutor(
        registry=registry
    )

    gateway = ScriptedGateway()

    runtime = ManualAgentLoop(
        gateway=gateway,
        registry=registry,
        executor=executor,
    )

    answer = await runtime.run(
        objective=(
            "Investigate onboarding-api."
        )
    )

    assert (
        answer
        == "onboarding-api is currently degraded."
    )

    assert len(
        gateway.requests
    ) == 2