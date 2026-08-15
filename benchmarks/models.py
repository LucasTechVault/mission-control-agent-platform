from pydantic import BaseModel, ConfigDict

class BenchmarkSample(BaseModel):
    """Result of 1 Mission Control inference request."""
    
    model_config = ConfigDict(extra="forbid")
    
    experiment: str
    run_id: int
    concurrency: int
    max_tokens: int
    
    succeeded: bool
    finish_reason: str | None = None
    error: str | None = None
    
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    ttft_ms: float | None = None
    e2e_latency_ms: float | None = None
    
    decode_duration_ms: float | None = None
    approx_tpot_ms: float | None = None
    
    decode_token_per_second: float | None = None
    e2e_token_per_second: float | None = None
    
class BenchmarkSummary(BaseModel):
    """Aggregate measurements for 1 benchmark workload (many BenchmarkSample)"""
    
    model_config = ConfigDict(extra="forbid")
    
    experiment: str
    concurrency: int # num concurrent
    requests: int # count
    successful_requests: int
    
    wall_time_seconds: float
    
    request_throughput_rps: float
    output_throughput_tokens_per_second: float
    
    # Tail Latency - track worst case, average hides outliers
    p50_ttft_ms: float | None
    p95_ttft_ms: float | None
    
    p50_e2e_latency_ms: float | None
    p95_e2e_latency_ms: float | None
    
    mean_decode_token_per_second: float | None
    