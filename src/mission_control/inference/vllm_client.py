import asyncio # timeout handling
import json
import time
from collections.abc import AsyncIterator

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from mission_control.config import Settings

from mission_control.inference.gateway import (
    ModelHTTPError,
    ModelResponseError,
    ModelTransportError,
    ModelTimeoutError
)
from mission_control.inference.requests import ModelRequest
from mission_control.inference.responses import (
    ModelResponse,
    ModelStreamEvent,
)

logger = structlog.get_logger(__name__)

# The letter - actual work AI wants to do
class _VLLMFunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    arguments: str

# The Envelope - wrapper that tracks request for system
class _VLLMToolCall(BaseModel):
    model_config=ConfigDict(extra="ignore")
    id: str
    type: str

    function: _VLLMFunctionCall

# '_' prefix specifies class for internal use only
# (The Letter)- handles the actual "meat" of the response
class _VLLMMessage(BaseModel):
    model_config = ConfigDict(extra="ignore") # extra fields are ignored, not forbidden
    
    content: str | None = None
    reasoning: str | None = None
    
    tool_calls: list[_VLLMToolCall] | None = None

# (The Envelope) - LLMs can generate multiple different answers to same prompt (choices)
# This class wraps the msg & adds a finish reason (success or failure)
class _VLLMChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    message: _VLLMMessage
    finish_reason: str | None = None

# (The Receipt) - Tracks token consumption
class _VLLMUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class _VLLMResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    model: str
    choices: list[_VLLMChoice]
    usage: _VLLMUsage
        
class VLLMModelGateway:
    """Model Gateway backed by vLLM OpenAI-compatible node."""
    
    # Dependency Injection
    def __init__(
        self, 
        settings: Settings, # URL, timeout rules, model name
        client: httpx.AsyncClient | None = None
        ) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient( # single, shared connection pool
            timeout=httpx.Timeout(
                connect=10.0, # connection cannot hang forever
                read=None, # we implementing Mission Control timeout ourselves
                write=10.0, # writes cannot hang forever
                pool=10.0 # pool acquisition cannot hang forever
            )
        )
        self._owns_client = client is None # use for clean up
    
    # timeout helper to resolve timeout duration
    # can be per-request, else from app default
    def _timeout_seconds(
        self,
        request: ModelRequest,
    ) -> float:
        return (
            request.timeout_seconds or
            self._settings.request_timeout_seconds
        )
    
    # centralize payload construction
    def _build_payload(
        self,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> dict:
        payload = {
            "model": self._settings.model_name,
            "messages": [
                msg.model_dump()
                for msg in request.messages # convert Pydantic Obj models to Plain dict for serdes
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": stream,
            "chat_template_kwargs": {
                "enable_thinking": request.enable_thinking
            },
        }
        
        # Handle domain contract response schema
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        request.response_schema_name or
                        "mission-control-response"
                    ),
                    "schema": request.response_schema,
                },
            }
        
        # M02-I04 - Ensure outbound requests (with tool call) perfectly formatted
        if request.tool_definitions:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.input_schema,
                        "strict": definition.strict
                    },
                } for definition in request.tool_definitions
            ]
            
            payload["tool_choice"] = request.tool_choice
        
        # Handle streaming
        if stream:
            payload["stream_options"] = {
                "include_usage": True # streaming OpenAI-compatible API usually omit token usage by default to save bandwidth
            }
        
        return payload
    
    async def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        
        # 1. Build Payload
        url = (
            f"{str(self._settings.inference_base_url).rstrip('/')}"
            "/chat/completions"
        )
        
        payload = self._build_payload(request, stream=False)
        timeout_seconds = self._timeout_seconds(request)
        
        logger.info(
            "model_request_started",
            request_id=request.request_id,
            model=self._settings.model_name,
            message_count=len(request.messages),
            max_tokens=request.max_tokens,
            enable_thinking=request.enable_thinking,
        )
                
        # 2. Send request
        started = time.perf_counter()
        
        try:
            # asyncio.timeout enforces network-level timeout
            async with asyncio.timeout(timeout_seconds):
                response = await self._client.post(
                    url,
                    json=payload,
                    headers={
                        "X-Request-ID": request.request_id,
                    },
                )

                response.raise_for_status()
                
        except TimeoutError as exc:
            raise ModelTimeoutError(
                f"Model request exceeded "
                f"{timeout_seconds:.2f}s"
                f"{request.request_id}"
            ) from exc
        
        # Catch underlying network / HTTPx inactivity timeouts
        except httpx.TimeoutException as exc:
            raise ModelTransportError(
                f"Model request timed out: {request.request_id}"
            ) from exc
        
        # Catch external cancellation signals
        except asyncio.CancelledError:
            logger.info(
                "model_request_cancelled",
                request_id=request.request_id,
                model=self._settings.model_name
            )
            # Cancellation is control flow. Do not swallow it
            raise

        except httpx.RequestError as exc:
            raise ModelTransportError(
                f"Unable to reach inference server: {exc}"
            ) from exc
        
        except httpx.HTTPStatusError as exc:
            raise ModelHTTPError(
                "Inference server returned "
                f"HTTP {exc.response.status_code} "
                f"{exc.response.text}"
            ) from exc
        
        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0
        
        # 3. Validation & Normalization of response
        try:
            payload = response.json()

            # Pydantic Object Validation
            raw = _VLLMResponse.model_validate(
                payload
            )

        except ValidationError as exc:
            logger.error(
                "model_response_validation_failed",
                request_id=request.request_id,
                validation_errors=exc.errors(),
                response_payload=payload,
            )

            raise ModelResponseError(
                f"Inference response failed validation: {exc}"
            ) from exc

        except ValueError as exc:
            logger.error(
                "model_response_json_failed",
                request_id=request.request_id,
                response_body=response.text,
            )

            raise ModelResponseError(
                f"Inference server returned invalid JSON: {exc}"
            ) from exc
        
        if not raw.choices:
            raise ModelResponseError(
                "Inference response contained no choices."
            )
            
        # 3.2 Normalize - Extract desired from raw response
        choice = raw.choices[0]
        
        normalized = ModelResponse(
            request_id=request.request_id,
            model=raw.model,
            text=choice.message.content,
            reasoning=choice.message.reasoning,
            finish_reason=choice.finish_reason,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            total_tokens=raw.usage.total_tokens,
            latency_ms=latency_ms
        )
        
        logger.info(
            "model_request_completed",
            request_id=request.request_id,
            model=normalized.model,
            finish_reason=normalized.finish_reason,
            prompt_tokens=normalized.prompt_tokens,
            completion_tokens=normalized.completion_tokens,
            total_tokens=normalized.total_tokens,
            latency_ms=round(normalized.latency_ms, 2),
        )
        
        return normalized
    
    # Implement Streaming
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        
        url = (
            f"{str(self._settings.inference_base_url).rstrip('/')}"
            "/chat/completions"
        )
        
        stream_payload = self._build_payload(
            request,
            stream=True,
        )
        
        timeout_seconds = self._timeout_seconds(request)
        
        started = time.perf_counter()
        
        first_token_ms: float | None = None
        finish_reason: str | None = None
        
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        
        model_name = self._settings.model_name
        
        logger.info(
            "model_stream_started",
            request_id=request.request_id,
            model=model_name,
            max_tokens=request.max_tokens,
            timeout_seconds=timeout_seconds
        )
        
        try:
            # Handle request timeout
            async with asyncio.timeout(timeout_seconds):
                async with self._client.stream(
                    "POST",
                    url,
                    json=stream_payload,
                    headers={
                        "X-Request-ID": request.request_id # uuid for distributed tracing
                    },
                ) as response:
                    if response.is_error:
                        await response.aread()
                        
                        raise ModelHTTPError(
                            "Inference server returned "
                            f"HTTP {response.status_code}: "
                            f"{response.text}"
                        )
                    
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        
                        data = line.removeprefix("data:").strip()
                        
                        # OpenAI-compatible SSE terminator
                        if data == "[DONE]":
                            elapsed_ms = (time.perf_counter() - started) * 1000.0
                            
                            logger.info(
                                "model_stream_completed",
                                request_id=request.request_id,
                                model=model_name,
                                finish_reason=finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                                latency_ms=round(elapsed_ms, 2,),
                                time_to_first_token=(round(first_token_ms, 2) if first_token_ms is not None else None),
                            )
                            
                            yield ModelStreamEvent(
                                request_id=request.request_id,
                                model=model_name,
                                finish_reason=finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                                elapsed_ms=elapsed_ms,
                                time_to_first_token_ms=first_token_ms,
                                done=True,
                            )
                            
                            return

                        try:
                            chunk = json.loads(data)
                        
                        except json.JSONDecodeError as exc:
                            raise ModelResponseError(
                                "Inference server returned an invalid streaming event."
                            ) from exc
                        
                        model_name = chunk.get(
                            "model",
                            model_name
                        )
                        
                        usage = chunk.get("usage")
                        
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens")
                            completion_tokens = usage.get("completion_tokens")
                            total_tokens = usage.get("total_tokens")
                        
                        choices = chunk.get("choices") or []
                        
                        # Usage-only chunks can have no choices
                        if not choices:
                            continue
                            
                        choice = choices[0]
                        
                        delta = choice.get("delta") or []
                        
                        text_delta = delta.get("content")
                        reasoning_delta = delta.get("reasoning")
                        
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        
                        if first_token_ms is None and (text_delta or reasoning_delta):
                            first_token_ms = (time.perf_counter() - started) * 1000.0
                        
                        # Don't emit meaningless empty chunks
                        if not (
                            text_delta or
                            reasoning_delta or
                            choice.get("finish_reason")
                        ):
                            continue
                            
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        
                        yield ModelStreamEvent(
                            request_id=request.request_id,
                            model=model_name,
                            text_delta=text_delta,
                            reasoning_delta=reasoning_delta,
                            finish_reason=choice.get("finish_reason"),
                            elapsed_ms=elapsed_ms,
                            time_to_first_token_ms=first_token_ms,
                            done=False,
                        )
        
        except TimeoutError as exc:
            logger.warning(
                "model_stream_time_out",
                request_id=request.request_id,
                timeout_seconds=timeout_seconds
            )          
            
            raise ModelTimeoutError(
                f"Streaming model request exceeded "
                f"{timeout_seconds:.2f}s: "
                f"{request.request_id}"
            ) from exc
        
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                f"Inference stream transport timed out:"
                f"{request.request_id}"
            ) from exc
        
        except asyncio.CancelledError:
            logger.info(
                "model_stream_cancelled",
                request_id=request.request_id,
                model=model_name,
            )
            
            raise
        
        except httpx.RequestError as exc:
            raise ModelTransportError(
                f"Inference stream failed: {exc}"
            ) from exc
                                    
    # Cleanup - Async Gateway holds network connection open.
    # Need to cleanly shut down when stopping backend server to prevent memory leaks
    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
