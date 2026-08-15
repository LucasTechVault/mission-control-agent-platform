import asyncio
import json
from datetime import datetime
from pathlib import Path

from mission_control.config import (
    get_settings,
)
from mission_control.inference.vllm_client import (
    VLLMModelGateway,
)

from benchmarks.runner import (
    run_workload,
)
from benchmarks.vllm_metrics import (
    poll_vllm_metrics,
)


RESULT_DIR = Path(
    "benchmark-results"
)

RESULT_DIR.mkdir(
    exist_ok=True
)


BENCHMARK_PROMPT = """
Generate a detailed technical analysis of possible
causes of an enterprise service outage.

Continue producing useful technical analysis until
the generation limit is reached.
""".strip()


async def main() -> None:

    settings = get_settings()

    print("=" * 70)
    print("MISSION CONTROL — M01-I10 INFERENCE BENCHMARK")
    print("=" * 70)

    print(
        f"Model    : {settings.model_name}"
    )
    print(
        f"Endpoint : {settings.inference_base_url}"
    )

    print()

    gateway = VLLMModelGateway(
        settings=settings
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    all_results = []

    try:
        # ----------------------------------
        # Warm-up
        # ----------------------------------

        print("Warm-up request...")

        await run_workload(
            gateway,
            experiment="warmup",
            prompt=BENCHMARK_PROMPT,
            concurrency=1,
            request_count=1,
            max_tokens=64,
            timeout_seconds=60,
        )

        print("Warm-up complete.\n")

        # ----------------------------------
        # A — Single request baseline
        # ----------------------------------

        workloads = [
            {
                "name": "baseline-c1",
                "concurrency": 1,
                "requests": 3,
                "max_tokens": 256,
            },

            # ----------------------------------
            # B — Output size
            # ----------------------------------

            {
                "name": "output-128",
                "concurrency": 1,
                "requests": 3,
                "max_tokens": 128,
            },
            {
                "name": "output-256",
                "concurrency": 1,
                "requests": 3,
                "max_tokens": 256,
            },
            {
                "name": "output-512",
                "concurrency": 1,
                "requests": 3,
                "max_tokens": 512,
            },

            # ----------------------------------
            # C — Concurrency
            # ----------------------------------

            {
                "name": "concurrency-2",
                "concurrency": 2,
                "requests": 4,
                "max_tokens": 256,
            },
            {
                "name": "concurrency-4",
                "concurrency": 4,
                "requests": 8,
                "max_tokens": 256,
            },
        ]

        for workload in workloads:

            print(
                f"Running {workload['name']}..."
            )

            stop_metrics = asyncio.Event()

            metrics_path = (
                RESULT_DIR
                / (
                    f"{timestamp}-"
                    f"{workload['name']}-"
                    "vllm.csv"
                )
            )

            metrics_task = asyncio.create_task(
                poll_vllm_metrics(
                    inference_base_url=str(
                        settings.inference_base_url
                    ),
                    stop_event=stop_metrics,
                    output_path=metrics_path,
                )
            )

            try:
                samples, summary = (
                    await run_workload(
                        gateway,
                        experiment=workload[
                            "name"
                        ],
                        prompt=BENCHMARK_PROMPT,
                        concurrency=workload[
                            "concurrency"
                        ],
                        request_count=workload[
                            "requests"
                        ],
                        max_tokens=workload[
                            "max_tokens"
                        ],
                        timeout_seconds=120,
                    )
                )

            finally:
                stop_metrics.set()

                await metrics_task

            result = {
                "summary": (
                    summary.model_dump()
                ),
                "samples": [
                    sample.model_dump()
                    for sample in samples
                ],
            }

            all_results.append(
                result
            )

            print(
                json.dumps(
                    summary.model_dump(),
                    indent=2,
                )
            )

            print()

        output_path = (
            RESULT_DIR
            / f"{timestamp}-i10.json"
        )

        output_path.write_text(
            json.dumps(
                all_results,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"Results written to "
            f"{output_path}"
        )

    finally:
        await gateway.aclose()


if __name__ == "__main__":
    asyncio.run(main())