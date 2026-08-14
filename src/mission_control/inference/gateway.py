from collections.abc import AsyncIterator
from typing import Protocol

from mission_control.inference.requests import ModelRequest
from mission_control.inference.responses import (
    ModelResponse,
    ModelStreamEvent,
)

class ModelGatewayError(RuntimeError):
    """Base exception for inference boundry failures."""
    
class ModelTransportError(ModelGatewayError):
    """The inference server could not be reached."""

class ModelHTTPError(ModelGatewayError):
    """The inference server returned an unsuccessful HTTP response."""

class ModelResponseError(ModelGatewayError):
    """The inference service returned an invalid response."""

class ModelTimeoutError(ModelGatewayError):
    """Model execution exceeded its allowed time budget."""

# Protocol means: Doesn't care what is passed into system / what class this inherits from
# Only cares if class has required method (e.g.'generate') and takes ModelRequest & returns ModelResponse
# Thus, we do not need explicit inheritance -> class Something(ModelGateway):
class ModelGateway(Protocol):
    """Capability Boundary for model inference."""
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generates 1 ModelResponse."""
    
    def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Streams 1 ModelStreamEvent"""
    

    