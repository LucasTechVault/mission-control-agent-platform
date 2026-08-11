import argparse
import asyncio

from mission_control.config import get_settings
from mission_control.inference.gateway import ModelGatewayError
from mission_control.inference.requests import ModelMessage, ModelRequest
from mission_control.inference.vllm_client import VLLMModelGateway

async def run_inference(prompt: str) -> None:
    """Send 1 prompt through Mission Control's inference boundary"""
    
    settings = get_settings()
    
    gateway = VLLMModelGateway(
        settings=settings
    )
    
    req = ModelRequest(
        messages=[
            ModelMessage(
                role="system",
                content=(
                    "You are the reasoning engine for Mission Control, "
                    "an enterprise investigation and decision platform"
                ),
            ),
            ModelMessage(
                role="user",
                content=prompt,
            ),
        ],
        temperature=0.2, # fairly deterministic
        top_p=0.8, # cum_prob > 0.8 for choices
        max_tokens=512,
        enable_thinking=False
    )
    
    try:
        res = await gateway.generate(req)
        
        print()
        print("=" * 60)
        print("MISSION CONTROL")
        print("=" * 60)
        
        print()
        print("Response:")
        print(res.text)
        
        if res.reasoning:
            print()
            print("Reasoning:")
            print(res.reasoning)
        
        print()
        print('-' * 60)
        print("Inference Metadata")
        print('-' * 60)
        
        print(f"Request ID       : {res.request_id}")
        print(f"Model            : {res.model}")
        print(f"Finish reason    : {res.finish_reason}")
        print(f"Prompt tokens    : {res.prompt_tokens}")
        print(f"Completion tokens: {res.completion_tokens}")
        print(f"Total tokens     : {res.total_tokens}")
        print(f"Latency          : {res.latency_ms:.2f} ms")
    
    except ModelGatewayError as exc:
        print()
        print("Mission Control Inference Failed")
        print(f"Reason: {exc}")
    
        raise SystemExit(1) from exc

    finally:
        await gateway.aclose()

def main() -> None:
    """Mission Control CLI entry-point"""
    parser = argparse.ArgumentParser(
        prog="mission-control",
        description=(
            "Send a request through Mission Control's "
            "local LLM Inference Boundary."
        )
    )
    
    parser.add_argument(
        "prompt",
        help="Prompt to send to the configured inference model.",
    )
    
    args = parser.parse_args()
    
    asyncio.run(
        run_inference(args.prompt)
    )

if __name__ == "__main__":
    main()
        