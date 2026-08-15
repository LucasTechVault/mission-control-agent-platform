from mission_control.tools.contracts import (
    ToolDefinition,
    ToolSideEffect
)

from mission_control.tools.schema import build_tool_definition

from mission_control.tools.investigation.service_health.models import (
    GetServiceHealthArguments, #ServiceHealth tool input Contract (LLM produce)
    ServiceHealthSnapshot # ServiceHealth tool Output Contract (tool produce)
)

from mission_control.tools.investigation.service_health.provider import (
    ServiceHealthProvider, # interface for ServiceHealth tool
)

# Tool seen by the model.
class GetServiceHealthTool:
    """_summary_
    Read-only Mission Control tool for retrieving normalized service health information.
    """
    
    NAME = "get_service_health"
    
    DESCRIPTION = (
        "Retrieve the current health of an application service in a deployment environment. "
        "Use this when investigating whether a service is healthy, degraded, or unhealthy."
    )
    
    def __init__(
        self,
        provider: ServiceHealthProvider, # dependency injection for desired provider
    ) -> None:
        self._provider = provider
        
        # Build LLM "menu" on the fly
        self._definition = build_tool_definition(
            name=self.NAME,
            description=self.DESCRIPTION,
            arguments_model=GetServiceHealthArguments,
            side_effect=ToolSideEffect.READ,
            strict=True
        )
    
    @property # definition of the tool, seen by LLM
    def definition(self) -> ToolDefinition:
        return self._definition
    
    @property # what the Python runtime will validate against
    def arguments_model(self) -> type[GetServiceHealthArguments]:
        return GetServiceHealthArguments
    
    async def execute(self, arguments: GetServiceHealthArguments) -> ServiceHealthSnapshot:
        return await (
            self._provider.get_service_health(
                service=arguments.service,
                environment=arguments.environment
            )
        )
    