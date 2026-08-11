import time

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from mission_control.config import Settings

from mission_control.inference.gateway import (
    ModelHTTPError,
    ModelResponseError,
    ModelTransportError
)
from mission_control.inference.requests import ModelRequest
from mission_control.inference.responses import ModelResponse

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
    
    msg: _VLLMMessage
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
    
    async def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        
        # 1. Build Payload
        url = (
            f"{str(self._settings.inference_base_url).rstrip('/')}"
            "/chat/completions"
        )
        
        payload = {
            "model": self._settings.model_name,
            "messages": [
                msg.model_dump() for msg in request.messages # convert Pydantic Obj models to Plain dict for serdes
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": request.enable_thinking
            }
        }
        
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
        
        try: # To handle network exceptions
            response = await self._client.post(
                url,
                json=payload,
                headers={
                    "X-Request-ID": request.request_id,
                },
            )

            response.raise_for_status()
        
        except httpx.TimeoutException as exc:
            raise ModelTransportError(
                f"Model request timed out: {request.request_id}"
            ) from exc

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
        
        # 3. Validation & Normalization of response - Gateway must validate allowed response
        try:
            
            # 3.1 - Validate using specified Pydantic model schema
            raw = _VLLMResponse.model_validate( 
                response.json()
            )
        
        except (ValueError, ValidationError) as exc:
            raise ModelResponseError(
                "Inference server returned an invalid response."
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
    
    # 4. Cleanup - Async Gateway holds network connection open.
    # Need to cleanly shut down when stopping backend server to prevent memory leaks
    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
