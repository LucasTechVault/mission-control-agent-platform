import asyncio
import statistics
import time
from collections.abc import Sequence

from mission_control.inference.gateway import (
    ModelGatewayError,
)

from mission_control.inference.requests import (
    ModelMessage,
    ModelRequest,
)

from mission_control.inference.vllm_client import (
    VLLMModelGateway,
)

from benchmarks.models import (
    BenchmarkSample,
    BenchmarkSummary,
)

# helper to calc tail latency
def percentile(
    values: Sequence[float],
    percentile_value: float,
) -> float | None:
    # Guard clause - empty arr
    if not values:
        return None
    
    # Guard clause - single element arr
    if len(values) == 1:
        return values[0]
    
    ordered = sorted(values)
    
    position = percentile_value / 100.0 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    
    frac = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac

# agent simulator
async def run_one(
    gateway: VLLMModelGateway,
    *,
    experiment: str,
    run_id: str,
    concurrency: int,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
) -> BenchmarkSample:
    
    request = ModelRequest(
        messages=[
            ModelMessage(
                role="system",
                content=(
                    "You are the inference engine for a systems benchmark. "
                    "Follow the user's instruction."
                ),
            ),
            ModelMessage(
                role="user",
                content=prompt
                ),
            ],
        temperature=0.2,
        top_p=0.8,
        max_tokens=max_tokens,
        enable_thinking=False,
        timeout_seconds=timeout_seconds,
    )
    
    started = time.perf_counter()
    
    # To handle streaming
    final_event = None
    
    try:
        async for event in gateway.stream(request):
            final_event = event
        
    except ModelGatewayError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        
        return BenchmarkSample(
            experiment=experiment,
            run_id=run_id,
            concurrency=concurrency,
            max_tokens=max_tokens,
            succeeded=False,
            error=str(exc),
            e2e_latency_ms=elapsed_ms
        )
    
    if final_event is None:
        return BenchmarkSample(
            experiment=experiment,
            run_id=run_id,
            concurrency=concurrency,
            max_tokens=max_tokens,
            succeeded=False,
            error="Stream completed without a final event.",
        )

    # Final event exist
    e2e_ms = final_event.elapsed_ms
    ttft_ms = final_event.time_to_first_token_ms
    
    completion_tokens = final_event.completion_tokens
    
    # Derived metrics variable declaration
    decode_duration_ms: float | None = None
    decode_tps: float | None = None
    tpot_ms: float | None = None
    e2e_tps: float | None = None
    
    if (
        ttft_ms is not None and
        e2e_ms > ttft_ms
    ):
        decode_duration_ms = e2e_ms - ttft_ms
    
    if (
        completion_tokens is not None and
        completion_tokens > 0 and
        e2e_ms > 0
    ):
        e2e_tps = completion_tokens / (e2e_ms / 1000.0)
    
    if (
        completion_tokens is not None and
        completion_tokens > 0 and
        decode_duration_ms is not None and
        decode_duration_ms > 0 
    ):
        decode_tps = completion_tokens / (decode_duration_ms / 1000.0)
        
    if (
        completion_tokens is not None and
        completion_tokens > 1 and
        decode_duration_ms is not None
    ):
        tpot_ms = decode_duration_ms / (completion_tokens - 1) # first token considered in TTFT
    
    return BenchmarkSample(
        experiment=experiment,
        run_id=run_id,
        concurrency=concurrency,
        max_tokens=max_tokens,
        succeeded=True,
        finish_reason=final_event.finish_reason,
        prompt_tokens=final_event.prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=final_event.total_tokens,
        ttft_ms=ttft_ms,
        e2e_latency_ms=e2e_ms,
        decode_duration_ms=decode_duration_ms,
        approx_tpot_ms=tpot_ms,
        decode_token_per_second=decode_tps,
        e2e_token_per_second=e2e_tps
    )

# Aggregate Benchmarking - simulator for Mission Control Platform
# Orchestrates off multiple "agents"
async def run_workload(
    gateway: VLLMModelGateway,
    *,
    experiment: str,
    prompt: str,
    concurrency: int,
    request_count: int,
    max_tokens: int,
    timeout_seconds: float,
) -> tuple[list[BenchmarkSample], BenchmarkSummary]:
    
    # create num locks to allow workers to use
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(run_id: int) -> BenchmarkSample:
        async with semaphore:
            return await run_one(
                gateway,
                experiment=experiment,
                run_id=run_id,
                concurrency=concurrency,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds
            )
    
    # Global stopwatch - total duration of entire stress test
    started = time.perf_counter()
    
    samples = await asyncio.gather(*[worker(run_id) for run_id in range(1, request_count + 1)])
    wall_seconds = time.perf_counter() - started
    
    successful = [sample for sample in samples if sample.succeeded]
    
    total_output_tokens = sum(sample.completion_tokens or 0 for sample in successful)
    ttfts = [sample.ttft_ms for sample in successful if sample.ttft_ms is not None]
    latencies = [sample.e2e_latency_ms for sample in successful if sample.e2e_latency_ms is not None]
    decode_rates = [sample.decode_token_per_second for sample in successful if sample.decode_token_per_second is not None]
    
    summary = BenchmarkSummary(
        experiment=experiment,
        concurrency=concurrency,
        requests=request_count,
        successful_requests=len(successful),
        wall_time_seconds=wall_seconds,
        request_throughput_rps=len(successful) / wall_seconds if wall_seconds > 0 else 0.0,
        output_throughput_tokens_per_second=total_output_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        p50_ttft_ms=percentile(ttfts, 50),
        p95_ttft_ms=percentile(ttfts, 95),
        p50_e2e_latency_ms=percentile(latencies, 50),
        p95_e2e_latency_ms=percentile(latencies, 95),
        mean_decode_token_per_second=statistics.mean(decode_rates if decode_rates else None),
    )
    
    return samples, summary
    