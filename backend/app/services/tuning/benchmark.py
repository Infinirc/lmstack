"""HTTP benchmark implementation for tuning agent.

This module contains the benchmark functions used to measure
model performance during auto-tuning.
"""

import asyncio
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)


async def run_http_benchmark(
    base_url: str,
    model_name: str = "default",
    num_requests: int = 20,
    concurrency: int = 4,
    input_tokens: int = 128,
    output_tokens: int = 64,
) -> dict:
    """Run actual HTTP benchmark against an OpenAI-compatible endpoint.

    Args:
        base_url: Base URL of the OpenAI-compatible API (e.g., "http://localhost:8000/v1")
        model_name: Model name to use for API calls
        num_requests: Number of requests to send
        concurrency: Number of concurrent requests
        input_tokens: Approximate input token count
        output_tokens: Max output tokens

    Returns:
        Dictionary with benchmark results including throughput and latency metrics
    """
    # Generate test prompt with approximate token count
    test_prompt = "Write a detailed explanation about " + " ".join(
        ["artificial intelligence"] * (input_tokens // 3)
    )

    results = []
    errors = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def make_request(client: httpx.AsyncClient) -> dict | None:
        nonlocal errors
        async with semaphore:
            start_time = time.perf_counter()
            first_token_time = None
            token_times = []
            total_tokens = 0

            try:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": test_prompt}],
                        "max_tokens": output_tokens,
                        "stream": True,
                    },
                    timeout=60.0,
                ) as response:
                    if response.status_code != 200:
                        errors += 1
                        return None

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                content = (
                                    chunk.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if content:
                                    current_time = time.perf_counter()
                                    if first_token_time is None:
                                        first_token_time = current_time
                                    token_times.append(current_time)
                                    total_tokens += 1
                            except json.JSONDecodeError:
                                pass

                end_time = time.perf_counter()

                if first_token_time and total_tokens > 0:
                    ttft = (first_token_time - start_time) * 1000  # ms
                    total_time = end_time - start_time

                    # Calculate TPOT (time per output token) excluding TTFT
                    if total_tokens > 1:
                        generation_time = end_time - first_token_time
                        tpot = (generation_time / (total_tokens - 1)) * 1000  # ms
                    else:
                        tpot = 0

                    return {
                        "ttft_ms": ttft,
                        "tpot_ms": tpot,
                        "total_tokens": total_tokens,
                        "total_time_s": total_time,
                    }
            except Exception as e:
                logger.warning(f"Benchmark request failed: {e}")
                errors += 1
                return None

    async with httpx.AsyncClient() as client:
        # Warm up with a few requests
        logger.info("Warming up benchmark endpoint...")
        for _ in range(min(2, num_requests)):
            await make_request(client)

        # Run actual benchmark
        logger.info(f"Running {num_requests} benchmark requests with concurrency {concurrency}...")
        tasks = [make_request(client) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    # Filter out failed requests
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        return {"success": False, "error": "All requests failed", "errors": errors}

    # Calculate metrics
    ttft_values = [r["ttft_ms"] for r in valid_results]
    tpot_values = [r["tpot_ms"] for r in valid_results if r["tpot_ms"] > 0]
    total_tokens = sum(r["total_tokens"] for r in valid_results)
    total_time = sum(r["total_time_s"] for r in valid_results)

    avg_ttft = sum(ttft_values) / len(ttft_values)
    avg_tpot = sum(tpot_values) / len(tpot_values) if tpot_values else 0
    throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "success": True,
        "metrics": {
            "throughput_tps": round(throughput, 2),
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_tpot_ms": round(avg_tpot, 2),
            "p50_ttft_ms": round(sorted(ttft_values)[len(ttft_values) // 2], 2),
            "p99_ttft_ms": (
                round(sorted(ttft_values)[int(len(ttft_values) * 0.99)], 2)
                if len(ttft_values) > 1
                else round(ttft_values[0], 2)
            ),
        },
        "summary": {
            "total_requests": num_requests,
            "successful_requests": len(valid_results),
            "failed_requests": errors,
            "total_tokens_generated": total_tokens,
        },
    }
