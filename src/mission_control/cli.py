import argparse
import asyncio

from mission_control.config import get_settings
from mission_control.inference.gateway import ModelGatewayError
from mission_control.inference.requests import ModelMessage, ModelRequest
from mission_control.inference.vllm_client import VLLMModelGateway

async def run_inference(
    prompt: str,
    *,
    stream: bool,
    timeout: float | None,
    max_tokens: int,
    ) -> None:
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
        max_tokens=max_tokens,
        enable_thinking=False,
        timeout_seconds=timeout
    )
    
    try:
        if stream:
            print()
            print("=" * 40)
            print("MISSION CONTROL - STREAMING")
            print("=" * 40)
            print()
            
            final_event = None
            
            async for event in gateway.stream(req):
                if event.text_delta:
                    print(event.text_delta, end="", flush=True)
                
                final_event = event
            
            print()
            print()
            
            if final_event is not None:
                print('-' * 40)
                print("Streaming Metadata")
                print('-' * 40)
                
                print(
                    f"Finish Reason:    : "
                    f"{final_event.finish_reason}"
                    )
                
                print(
                    f"Prompt Tokens     : "
                    f"{final_event.prompt_tokens}"
                )
                
                print(
                    f"Completion Tokens     : "
                    f"{final_event.completion_tokens}"
                )
                                
                print(
                    f"Total Tokens     : "
                    f"{final_event.total_tokens}"
                )
                
                print(
                    f"Time to first token: "
                    f"{final_event.time_to_first_token_ms:.2f} ms"
                    if final_event.time_to_first_token_ms is not None
                    else "Time to first token: Unknown"
                )
                
                print(
                    f"Total Latency     :"
                    f"{final_event.elapsed_ms:.2f} ms"
                )
        else: # Non-streaming output
            res = await gateway.generate(req)
                
            print()
            print("=" * 60)
            print("MISSION CONTROL - HTTP REQUEST")
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
    
    # Parsers added after M01-I09 Streaming Impl
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated output as it arrives"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Total inference deadline in seconds."
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens."
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(
            run_inference(
                    args.prompt,
                    stream=args.stream,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    )
        )
        
    except KeyboardInterrupt:
        print()
        print("Mission Control request cancelled.")


if __name__ == "__main__":
    main()
        