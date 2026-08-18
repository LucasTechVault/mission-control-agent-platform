import structlog

from mission_control.inference.gateway import ModelGateway
from mission_control.tools.registry import ToolRegistry
from mission_control.tools.executor import ToolExecutor
from mission_control.inference.requests import (
    ModelMessage,
    ModelRequest,
)

logger = structlog.get_logger(__name__)

class AgentLoopError(RuntimeError):
    """Manual Mission Control agent loop failed."""

SYSTEM_PROMPT = """
You are Mission Control, an enterprise investigation agent.

Your job is to gather evidence using the available read-only investigation tools and answer the user's objective

RuleS:
- Use tools when evidence is required.
- Never invent tool results.
- After receiving a tool result, decide whether more evidence is required.
- Do not repeat the same tool call unless new information justifies doing so.
- When enough evidence has been gathered, stop calling tools and provide a concise final answer grounded in observations.
""".strip()
    
class ManualAgentLoop:
    """
    First framework-free Mission Control agent runtime.
    
    The runtime owns the loop.
    The model chooses whether to:
    - request tools
    - provide a final answer
    
    This implementation deliberately avoids AgentState and orchestration frameworks.
    """
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        registry: ToolRegistry,
        executor: ToolExecutor,
        model_max_tokens: int = 512,
        model_timeout_seconds: float = 60.0,
        
        # Temporary emergency protection for this exercise
        emergency_max_model_turns: int = 8
    ) -> None:
        self._gateway = gateway
        self._registry = registry
        self._executor = executor
        
        self._model_max_tokens = model_max_tokens
        self._model_timeout_seconds = model_timeout_seconds
        self._emergency_max_model_turns = emergency_max_model_turns
        
    async def run(self, objective: str) -> str:
        if not objective.strip():
            raise ValueError("Agent objective cannot be empty.")
    
        # This list will be current implicit runtime state.
        # This is insufficient and will require AgentState
        messages: list[ModelMessage] = [
            ModelMessage(
                role="system",
                content=SYSTEM_PROMPT
            ),
            ModelMessage(
                role="user",
                content=objective
            )
        ]
        logger.info("agent_run_started", objective=objective)
        
        # ==============================
        # The Manual Agent Loop
        # ==============================
        for model_turn in range(1, self._emergency_max_model_turns + 1):
            logger.info(
                "agent_model_turn_started",
                model_turn=model_turn,
                message_count=len(messages)
            )
            
            # 1. Build ModelRequest, in accordance to vLLM expectations
            request = ModelRequest(
                messages=messages,
                tool_definitions=self._registry.definitions(), # provide menu to LLM
                tool_choice="auto", # model may call tool OR finish
                
                temperature=0.0, # deterministic 
                top_p=1.0, # cum prob up to 1.0
                max_tokens=self._model_max_tokens,
                enable_thinking=False, # Print internal monologue or not
                timeout_seconds=self._model_timeout_seconds
            )
            
            response = await self._gateway.generate(request)
            
            # 2 Handle Action (ToolCall) Proposal
            if response.tool_calls:
                logger.info(
                    "agent_actions_requested",
                    model_turn=model_turn,
                    tool_count=len(response.tool_calls),
                    tool_names=[call.tool_name for call in response.tool_calls]
                )
                
                # 2.1 Preserve model's action proposal in history
                messages.append(
                    ModelMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls
                    )
                )
                
                # 2.2 ACT + OBSERVE
                for tool_call in response.tool_calls:
                    
                    # ACT
                    tool_result = await self._executor.execute(tool_call) # executor will perform validation of tool
                    
                    logger.info(
                        "agent_observation_received",
                        model_turn=model_turn,
                        call_id=tool_call.call_id,
                        tool_name=tool_call.tool_name,
                        status=tool_result.status
                    )
                    
                    # OBSERVE
                    messages.append(
                        ModelMessage(
                            role="tool",
                            tool_call_id=tool_call.call_id,
                            content=tool_result.model_dump_json()
                        )
                    )
                    
                # Finished executing ALL tools requested during model turn, rerun prompt but with context information
                continue
                
            # 3 - response is not toolcall but final response
            if response.text and response.text.strip():
                logger.info(
                    "agent_run_completed",
                    model_turn=model_turn,
                    finish_reason=response.finish_reason
                )

                return response.text

            # Handle scenario where model return neither tool_call nor final response
            raise AgentLoopError("Model returned neither tool calls nor a final answer.")
        
        # Outside for loop, loop exhausted.
        raise AgentLoopError(f"Emergency model-turn ceiling {self._emergency_max_model_turns} was exceeded.")