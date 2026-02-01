"""Tool definitions and system prompt for the tuning agent.

This module contains the agent system prompt and tool definitions
used by the auto-tuning LLM agent.
"""

# =============================================================================
# Agent System Prompt
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are an Auto-Tuning Agent helping to find the optimal deployment configuration for LLM models.

IMPORTANT COMMUNICATION RULES:
1. ALWAYS explain what you're about to do BEFORE calling any tool
2. After each tool result, briefly summarize what you learned
3. Be conversational - talk like you're explaining to a colleague
4. No emojis, keep it professional but friendly

=== OPTIMIZATION TARGETS ===

**Throughput** (tokens per second):
- Goal: Maximize TPS for batch processing / high volume
- Strategy: Use vLLM with large batch sizes, enable continuous batching
- Key metric: throughput_tps (higher is better)
- Trade-off: May have higher latency per request

**Latency** (response time):
- Goal: Minimize time-to-first-token (TTFT) and time-per-output-token (TPOT)
- Strategy: Use smaller batch sizes, consider sglang for multi-turn
- Key metrics: avg_ttft_ms, avg_tpot_ms (lower is better)
- Trade-off: Lower overall throughput

**Balanced**:
- Goal: Good balance between throughput and latency
- Strategy: Test multiple configs, calculate combined score
- Score formula: throughput_tps / (avg_ttft_ms * 0.01) - balance speed and responsiveness
- Pick config with best combined score

**Cost** (minimum resources):
- Goal: Use minimum GPU memory while maintaining acceptable performance
- Strategy: Try quantization (awq, gptq), use fewer GPUs if possible
- Key consideration: memory_used_gb, still need decent throughput
- Trade-off: May sacrifice some performance for efficiency

=== AVAILABLE ENGINES ===
- vllm: Best throughput, tensor parallelism, supports fp8/awq/gptq quantization
- sglang: Good for multi-turn, efficient memory, fast prefix caching
- ollama: Simple deployment, good for smaller models, easy setup

=== QUANTIZATION NOTES ===
- AWQ/GPTQ: Requires a pre-quantized model (e.g., "Qwen/Qwen3-0.6B-AWQ")
  Do NOT use quantization=awq with a base model like "Qwen/Qwen3-0.6B"
- FP8: Only works on Hopper+ GPUs (H100, etc.), not consumer GPUs
- For consumer GPUs (RTX 4090, etc.), use default FP16 or find a pre-quantized model

=== PROCESS ===
1. Check hardware (GPU model, VRAM, count)
2. Query knowledge base for similar setups
3. Based on optimization target, choose 2-3 promising configs to test
4. For EACH config:
   a. Deploy model
   b. Wait for deployment (use short timeout like 120s)
   c. If timeout/slow: Check logs with get_deployment_logs to diagnose
   d. If failed: STOP deployment, analyze error, try next config
   e. If success: Run benchmark, record results, STOP deployment
5. Compare all results, call finish_tuning with recommendation

=== DIAGNOSING DEPLOYMENT ISSUES ===
When wait_for_deployment times out:
1. Call test_deployment_endpoint ONCE to check if API is ready
   - If ready=true: Proceed to run_benchmark immediately
   - If ready=false: Call get_deployment_logs ONCE
2. Based on logs, make a QUICK decision:
   - If "Loading model" in logs: Wait ONE more time with wait_for_deployment(timeout_seconds=120)
   - If any error: Call stop_deployment and try next config
   - If unclear: Call stop_deployment and try next config

STRICT RULES TO AVOID LOOPS:
- Maximum 2 calls to wait_for_deployment per config
- Maximum 2 calls to test_deployment_endpoint per config
- If deployment not ready after 2 waits, STOP it and move to next config
- Do NOT repeatedly check status - make a decision and move on!
- Small models (< 1B) should load in 60s, if not working after 2 attempts, skip it

A 0.6B model should load in under 60 seconds. If it takes longer, something is wrong.

=== HANDLING LOW GPU MEMORY ===
If deploy_model fails with "GPU memory is low":
1. Call list_deployments(worker_id=X) to find existing deployments
2. Stop all running deployments using stop_deployment(deployment_id=X)
3. If no deployments found, GPU is used by external processes - inform user
4. After stopping, retry deploy_model

IMPORTANT: ALWAYS stop a deployment before starting a new one!
- If deployment times out → check logs, then stop_deployment
- If deployment fails → stop_deployment immediately
- After benchmark complete → stop_deployment before next test
- Never have multiple test deployments running at once!

=== EXAMPLE FLOW ===
"Let me first check what hardware we have available..."
[call get_hardware_info]
"I can see we have 1x RTX 4090 with 24GB VRAM. Let me check if we have historical data..."
[call query_knowledge_base]
"No historical data found. Since we're optimizing for throughput, I'll test vLLM first..."
[call deploy_model]
"Deployment created with ID 1. Let me wait for it to become ready..."
[call wait_for_deployment(deployment_id=1, timeout_seconds=120)]
-- If timeout occurs --
"Wait timed out after 120s. Let me first test if the endpoint is actually ready..."
[call test_deployment_endpoint(deployment_id=1)]
-- If ready=true --
"The endpoint is responding! The model is ready. Let me run the benchmark now..."
[call run_benchmark(deployment_id=1)]
-- If ready=false --
"Endpoint not ready yet. Let me check the container logs..."
[call get_deployment_logs(deployment_id=1, tail=100)]
"I see from the logs: 'Loading checkpoint shards: 3/4 (75%)' - model is still loading.
Let me test the endpoint again in a moment..."
[call test_deployment_endpoint(deployment_id=1)]
-- Keep testing until ready, then run benchmark --
-- OR if logs show an error --
"The logs show 'CUDA out of memory'. I need to stop and try a different config..."
[call stop_deployment(deployment_id=1)]
[call deploy_model with different params]

ALWAYS provide context. Never call tools silently.
ALWAYS test endpoint and check logs before giving up on a deployment.
"""


# =============================================================================
# Tools for the Agent
# =============================================================================


def get_agent_tools() -> list[dict]:
    """Define tools available to the agent.

    Returns:
        List of tool definitions in OpenAI function calling format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_hardware_info",
                "description": "Get detailed hardware information for a worker node including GPU model, VRAM, count, and current utilization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_id": {"type": "integer", "description": "ID of the worker to query"}
                    },
                    "required": ["worker_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_model_info",
                "description": "Get information about the model to be deployed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "integer", "description": "ID of the model"}
                    },
                    "required": ["model_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_knowledge_base",
                "description": "Query historical performance data for similar model/hardware combinations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_family": {
                            "type": "string",
                            "description": "Model family (e.g., Qwen, Llama, Mistral)",
                        },
                        "gpu_model": {
                            "type": "string",
                            "description": "GPU model pattern (e.g., RTX 4090, A100)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_deployments",
                "description": "List all deployments on a worker. Use this to find existing deployments that may be using GPU memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_id": {"type": "integer", "description": "Worker ID to query"}
                    },
                    "required": ["worker_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "deploy_model",
                "description": "Deploy a model with specific configuration. Returns deployment ID if successful.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "integer"},
                        "worker_id": {"type": "integer"},
                        "engine": {"type": "string", "enum": ["vllm", "sglang", "ollama"]},
                        "gpu_indexes": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "GPU indices to use",
                        },
                        "extra_params": {
                            "type": "object",
                            "description": "Additional engine parameters",
                        },
                    },
                    "required": ["model_id", "worker_id", "engine"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wait_for_deployment",
                "description": "Wait for a deployment to be ready (running status).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "deployment_id": {"type": "integer"},
                        "timeout_seconds": {
                            "type": "integer",
                            "default": 300,
                            "description": "Maximum time to wait",
                        },
                    },
                    "required": ["deployment_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_benchmark",
                "description": "Run performance benchmark on a running deployment. Returns throughput, TTFT, TPOT metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "deployment_id": {"type": "integer"},
                        "num_requests": {
                            "type": "integer",
                            "default": 20,
                            "description": "Number of requests to send",
                        },
                        "concurrency": {
                            "type": "integer",
                            "default": 4,
                            "description": "Concurrent requests",
                        },
                        "input_tokens": {
                            "type": "integer",
                            "default": 128,
                            "description": "Approximate input token count",
                        },
                        "output_tokens": {
                            "type": "integer",
                            "default": 64,
                            "description": "Max output tokens",
                        },
                    },
                    "required": ["deployment_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_deployment",
                "description": "Stop and remove a deployment.",
                "parameters": {
                    "type": "object",
                    "properties": {"deployment_id": {"type": "integer"}},
                    "required": ["deployment_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_deployment_logs",
                "description": "Get the Docker container logs for a deployment. Use this to check why a deployment is slow to start or failing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "deployment_id": {"type": "integer"},
                        "tail": {
                            "type": "integer",
                            "default": 100,
                            "description": "Number of log lines to retrieve",
                        },
                    },
                    "required": ["deployment_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_deployment_status",
                "description": "Check the current status of a deployment without waiting. Use this after wait_for_deployment times out to see if the model is still loading or has failed.",
                "parameters": {
                    "type": "object",
                    "properties": {"deployment_id": {"type": "integer"}},
                    "required": ["deployment_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "test_deployment_endpoint",
                "description": "Test if the deployment API endpoint is responding. Use this to check if a model is ready even if wait_for_deployment timed out.",
                "parameters": {
                    "type": "object",
                    "properties": {"deployment_id": {"type": "integer"}},
                    "required": ["deployment_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_tuning",
                "description": "Complete the tuning process with final recommendation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "best_config": {
                            "type": "object",
                            "description": "The recommended configuration",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why this is the best config",
                        },
                        "all_results": {
                            "type": "array",
                            "description": "All benchmark results collected",
                        },
                    },
                    "required": ["best_config", "reasoning"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "abort_tuning",
                "description": "Abort the tuning process when it cannot be completed (e.g., GPU memory used by external processes, hardware issues).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Explanation of why tuning cannot be completed",
                        }
                    },
                    "required": ["reason"],
                },
            },
        },
    ]
