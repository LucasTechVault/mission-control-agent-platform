import asyncio
import time

import structlog
from pydantic import BaseModel, ValidationError

from mission_control.tools.contracts import (
    ToolCall,
    ToolError,
    ToolExecutionStatus,
    ToolResult,
)

from mission_control.tools.registry import (
    ToolNotFoundError,
    ToolRegistry
)

logger = structlog.get_logger(__name__)

class ToolExecutor:
    """
    Deterministic execution boundary:
    Between model-proposed ToolCalls & registered Mission Control capabilities.
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        default_timeout_seconds: float = 10.0,
    ) -> None:
        if default_timeout_seconds <= 0:
            raise ValueError("Tool timeout must be positive.")
        self._registry = registry
        self._default_timeout_seconds = default_timeout_seconds
    
    # Concrete implementation of generic `execute`
    # generic = all tools will call this (provider will perform specific impl)
    async def execute(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds
        
        logger.info(
            "tool_execution_started",
            call_id=call.call_id,
            tool_name=call.tool_name,
            parent_request_id=call.parent_request_id
        )
        
        # Capability Allowlist check
        try:
            tool = self._registry.get(call.tool_name)
        
        except ToolNotFoundError:
            return self._error_result(
                call=call,
                started=started,
                code="tool_not_found",
                message=f"Tool {call.tool_name!r} is not registered.",
                retryable=False,
            )
        
        # Deterministic argument validation
        try:
            arguments = tool.arguments_model.model_validate(call.arguments)
        except ValidationError as exc:
            return self._error_result(
                call=call,
                started=started,
                code="invalid_arguments",
                message=str(exc),
                retryable=False
            )
        
        # Controlled Tool Execution
        try:
            async with asyncio.timeout(timeout): # to handle possible timeout by provider
                output = await tool.execute(arguments) # async for multi-tool execution
        except TimeoutError:
            return self._error_result( # Timeout error normalization
                call=call,
                started=started,
                code="tool_timeout",
                message=f"Tool execution exceeded. {timeout:.2f}s",
                retryable=True,
            )
        except asyncio.CancelledError:
            logger.info(
                "tool_execution_cancelled",
                call_id=call.call_id,
                tool_name=call.tool_name,
            )
            
            raise # Cancellation remains control flow
            
        except Exception as exc:
            logger.info(
                "tool_execution_failed",
                call_id=call.call_id,
                tool_name=call.tool_name,
            )
                        
            return self._error_result( # Generic execution failure error normalization
                call=call,
                started=started,
                code="tool_execution_failed",
                message=str(exc),
                retryable=False
            )
        
        # Normalize successful output
        duration_ms = (time.perf_counter() - started) * 1000.0
        
        if isinstance(output, BaseModel):
            normalized_output = output.model_dump(mode="json")
        else:
            normalized_output = output
        
        result = ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            output=normalized_output,
            duration_ms=duration_ms
        )
        
        logger.info(
            "tool_execution_completed",
            call_id=call.call_id,
            tool_name=call.tool_name,
            duration_ms=round(duration_ms, 2)
        )
        
        return result

    # Error normalization
    @staticmethod
    def _error_result(
        *,
        call: ToolCall,
        started: float,
        code: str,
        message: str,
        retryable: bool,
    ) -> ToolResult:
        duration_ms = (time.perf_counter() - started) * 1000.0
        
        # Error itself is a result in LLM loop
        # Let agent reason about invalid arguments / invalid execution
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolExecutionStatus.ERROR,
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
            ),
            duration_ms=duration_ms
        )
            