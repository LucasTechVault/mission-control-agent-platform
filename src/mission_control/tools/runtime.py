from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mission_control.inference.requests import (
    ModelMessage,
    ModelRequest,
)
from mission_control.inference.vllm_client import (
    VLLMModelGateway,
)
from mission_control.tools.contracts import (
    ToolCall,
    ToolResult,
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


class ToolRuntimeError(RuntimeError):
    """Mission Control could not complete a model-requested tool execution."""


class ToolExecutionTrace(BaseModel):
    """
    Result of one complete model → tool execution boundary.

    This deliberately stops after deterministic execution.
    It is not yet an agent loop.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str
    model: str

    tool_call: ToolCall
    tool_result: ToolResult


def build_default_tool_registry() -> ToolRegistry:
    """
    Build the capabilities available to the current local
    Mission Control development runtime.
    """

    provider = JsonServiceHealthProvider(
        Path(
            "configs/tools/service_health.json"
        )
    )

    health_tool = GetServiceHealthTool(
        provider=provider
    )

    registry = ToolRegistry()

    registry.register(
        health_tool
    )

    return registry


async def execute_one_tool_request(
    *,
    gateway: VLLMModelGateway,
    prompt: str,
    timeout_seconds: float = 60.0,
) -> ToolExecutionTrace:
    """
    Ask the model to request exactly one registered tool,
    then execute that proposal through Mission Control's
    deterministic execution boundary.

    This function intentionally performs only ONE model turn
    and ONE tool execution.
    """

    # --------------------------------------------------
    # 1. Construct Mission Control's capability allowlist
    # --------------------------------------------------

    registry = build_default_tool_registry()

    executor = ToolExecutor(
        registry=registry,
        default_timeout_seconds=10.0,
    )

    # --------------------------------------------------
    # 2. Ask the probabilistic model for a tool request
    # --------------------------------------------------

    request = ModelRequest(
        messages=[
            ModelMessage(
                role="system",
                content=(
                    "You are Mission Control, an enterprise "
                    "investigation system. "
                    "Use the available tool to satisfy the "
                    "user's investigation request. "
                    "Do not invent tool arguments."
                ),
            ),
            ModelMessage(
                role="user",
                content=prompt,
            ),
        ],

        # Definitions come from OUR registry.
        # The model does not invent available capabilities.
        tool_definitions=registry.definitions(),

        # For this integration test, require a tool call.
        tool_choice="required",

        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        enable_thinking=False,
        timeout_seconds=timeout_seconds,
    )

    model_response = await gateway.generate(
        request
    )

    # --------------------------------------------------
    # 3. Enforce one-shot runtime semantics
    # --------------------------------------------------

    if not model_response.tool_calls:
        raise ToolRuntimeError(
            "Model returned no tool call."
        )

    if len(model_response.tool_calls) != 1:
        raise ToolRuntimeError(
            "Expected exactly one tool call, "
            f"received {len(model_response.tool_calls)}."
        )

    tool_call = model_response.tool_calls[0]

    # --------------------------------------------------
    # 4. CROSS THE TRUST BOUNDARY
    #
    # Nothing above this point has executed the tool.
    # tool_call is only DATA proposed by the LLM.
    # --------------------------------------------------

    tool_result = await executor.execute(
        tool_call
    )

    # --------------------------------------------------
    # 5. Return a complete trace of this one execution
    # --------------------------------------------------

    return ToolExecutionTrace(
        request_id=request.request_id,
        model=model_response.model,
        tool_call=tool_call,
        tool_result=tool_result,
    )