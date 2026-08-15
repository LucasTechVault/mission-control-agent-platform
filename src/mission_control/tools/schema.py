from pydantic import BaseModel

from mission_control.tools.contracts import (
    ToolArguments,
    ToolDefinition,
    ToolSideEffect,
)

# Passed to vLLM - like a menu for the model.
# inject JSON schema into `tools` array of vLLM http payload
def build_tool_definition(
    *,
    name: str,
    description: str,
    arguments_model: type[ToolArguments],
    side_effect: ToolSideEffect = ToolSideEffect.NONE,
    strict: bool = True,
) -> ToolDefinition:
    """_summary_
    Build a model-facing ToolDefinition from the same Pydantic model Mission Control will use later for validation
    """
    
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=arguments_model.model_json_schema(),
        side_effect=side_effect,
        strict=strict
    )

# Executed by Mission Control
def validate_tool_arguments[TArguments: ToolArguments] (
    *,
    arguments_model: type[TArguments],
    raw_arguments: dict,
) -> TArguments:
    """_summary_
    Validate raw model-proposed arguments against the tool's typed Python contract.
    """
    return arguments_model.model_validate(raw_arguments)
