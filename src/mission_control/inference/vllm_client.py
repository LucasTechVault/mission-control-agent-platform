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

# '_' prefix specifies class for internal use only
# (The Letter)- handles the actual "meat" of the response
class _VLLMMessage(BaseModel):
    model_config = ConfigDict(extra="ignore") # extra fields are ignored, not forbidden
    
    content: str | None = None
    reasoning: str | None = None

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
            timeout=settings.request_timeout_seconds
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
    
    # Cleanup - Async Gateway holds network connection open.
    # Need to cleanly shut down when stopping backend server to prevent memory leaks
    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
