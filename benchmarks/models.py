from pydantic import BaseModel, ConfigDict

class BenchmarkSample(BaseModel):
    """Result of 1 Mission Control inference request."""
    
    model_config = ConfigDict(extra="forbid")
    
    experiment: str
    run_id: int
    concurrency: int
    