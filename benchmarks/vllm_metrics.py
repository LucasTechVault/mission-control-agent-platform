import asyncio
import re
import time
from pathlib import Path

import httpx

_METRIC_PATTERN = re.compile(
    r"^(vllm:"
    r"(?:kv_cache_usage_perc"
    r"|num_requests_running"
    r"|num_requests_waiting)"
    r")"
    r"(?:\{[^}]*\})?"
    r"\s+"
    r"([-+0-9.eE]+)$"
)

def metrics_url(inference_base_url: str, ) -> str:

    base = inference_base_url.rstrip('/')
    
    if base.endswith("/v1"):
        base = base[:-3]
        
    return f"{base}/metrics"

# 1. Prometheus-like scraper
def parse_selected_metrics(body: str,) -> dict[str, float]:
    
    values: dict[str, list[float]] = {}
    
    for line in body.splitlines():
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
            
        match = _METRIC_PATTERN.match(line)
        
        if match is None:
            continue
            
        name = match.group(1)
        val = float(match.group(2))
        
        values.setdefault(name, []).append(val)
        
        # Handles multi-GPU workloads
        # average max() or sum() accordingly
        return {
            "kv_cache_usage_perc": max(values.get("vllm:kv_cache_usage_perc", [0.0])),
            "num_requests_running": sum(values.get("vllm:num_requests_running", [0.0])),
            "num_requests_waiting": sum(values.get("vllm:num_requests_waiting", [0.0]))
        }

# Background x-ray loop - async alongside benchmark
async def poll_vllm_metrics(
    *,
    inference_base_url: str,
    stop_event: asyncio.Event,
    output_path: Path,
    interval_seconds: float = 0.5,
) -> None:
    url = metrics_url(inference_base_url)
    
    started = time.perf_counter()
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        with output_path.open('w', encoding="utf-8",) as file:
            file.write(
                "elapsed_seconds,"
                "kv_cache_usage_perc,"
                "num_requests_running,"
                "num_requests_waiting\n"
            )
            
            while not stop_event.is_set():
                try:
                    res = await client.get(url)
                    res.raise_for_status()
                    metrics = parse_selected_metrics(res.text)
                    elapsed = time.perf_counter() - started
                    
                    file.write(
                        f"{elapsed:.3f},"
                        f"{metrics['kv_cache_usage_perc']:.6f},"
                        f"{metrics['num_requests_running']:.0f},"
                        f"{metrics['num_requests_waiting']:.0f}\n"
                    )
                    
                    file.flush()
                
                except httpx.HTTPError:
                    # Benchmark should still run even if vLLM build does not expose metrics
                    return

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                except TimeoutError:
                    pass # skip if network timeout
                