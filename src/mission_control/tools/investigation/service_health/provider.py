import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from mission_control.tools.investigation.service_health.models import (
    Environment,
    ServiceHealthSnapshot,
)

class ServiceHealthProviderError(RuntimeError):
    """Base provider failure."""

class ServiceHealthRecordNotFoundError(ServiceHealthProviderError):
    """Requested service/environment does not exist."""

class ServiceHealthDataError(ServiceHealthProviderError):
    """Backing health data is malformed."""

# Port for ServiceHealth tool - The socket that dictates any data-fetching implementations must have:
    # 1. async def get_service_health
    # 2. returns ServiceHealthSnapshot
class ServiceHealthProvider(Protocol):
    """_summary_
    Port used by GetServiceHealthTool.
    
    Production implementations might query a monitoring platform
    Local development can use another adapter.
    """
    
    async def get_service_health(
        self,
        *,
        service: str,
        environment: Environment,
    ) -> ServiceHealthSnapshot:
        ...
    
# Helper class for Mock DB (JSON file)
class _ServiceHealthFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    services: list[ServiceHealthSnapshot]

# 1 implementation of ServiceHealthProvider
# Also known as adapter while ServiceHealthProvider is the socket
# - contains async def get_service_health & returns ServiceHealthSnapshot (all checks out with ServiceHealthProvider Protocol)
class JsonServiceHealthProvider:
    """_summary_
    Local development adapter for tool-testing, backed by JSON file.
    
    The file is loaded once during construction and thereafter exposed through a read-only lookup interface.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records = self._load_records(path)
    
    @staticmethod
    def _load_records(path: Path) -> dict[tuple[str, str], ServiceHealthSnapshot]:
        try:
            raw_text = path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            fixture = _ServiceHealthFixture.model_validate(raw_data) 
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ServiceHealthDataError(f"Could not load service health date from {path}")
        
        records: dict[tuple[str, str], ServiceHealthSnapshot] = {}
        
        for snapshot in fixture.services:
            key = (
                snapshot.service,
                snapshot.environment
            )
            
            if key in records:
                raise ServiceHealthDataError(f"Duplicate service health record: {key}")

            records[key] = snapshot
        
        return records
    
    async def get_service_health(
        self,
        *,
        service: str,
        environment: Environment
    ) -> ServiceHealthSnapshot:
        key = (
            service,
            environment
        )
        
        snapshot = self._records.get(key)
        
        if snapshot is None:
            raise ServiceHealthRecordNotFoundError(f"No service health record for service={service!r}, environment={environment!r}")
        
        return snapshot.model_copy(deep=True)
            