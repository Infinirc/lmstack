"""
Auto-Tuning Agent Service

A true LLM-driven agent that:
1. Uses an LLM to reason about configurations
2. Actually deploys models with different configs
3. Runs real benchmarks against deployed endpoints
4. Analyzes results and decides next steps
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime

import httpx
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import async_session_maker
from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import LLMModel
from app.models.tuning import PerformanceKnowledge, TuningJob, TuningJobStatus
from app.models.worker import Worker

logger = logging.getLogger(__name__)


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
1. FIRST call test_deployment_endpoint to check if API is actually ready
   - If ready=true: Great! Proceed to run_benchmark
   - If ready=false: Continue to step 2
2. Call get_deployment_logs to check container logs (use tail=100 or more)
3. Look for common patterns in logs:
   - "Loading checkpoint shards" or "Loading model weights" - model is loading, keep waiting
   - "INFO: Started server process" or "Uvicorn running" - vLLM is ready!
   - "CUDA out of memory" - try quantization or fewer GPUs
   - "Error" or "Exception" - check the error message
4. Based on logs, decide:
   - If model loading: call test_deployment_endpoint every 30s until ready
   - If OOM error: stop_deployment and try with quantization
   - If other error: stop_deployment and try different engine/config

DO NOT just give up on timeout - always test endpoint and check logs first!
A 0.6B model should load in 1-2 minutes, larger models (7B+) may take 5-10 minutes.

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
    """Define tools available to the agent"""
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


# =============================================================================
# Tool Implementations
# =============================================================================


class AgentToolExecutor:
    """Execute agent tools with real system interactions"""

    def __init__(self, db: AsyncSession, job: TuningJob):
        self.db = db
        self.job = job
        self.created_deployments: list[int] = []

    async def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return result as string"""
        try:
            method = getattr(self, f"_tool_{tool_name}", None)
            if method:
                result = await method(**args)
                return json.dumps(result, indent=2, default=str)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    async def _tool_get_hardware_info(self, worker_id: int) -> dict:
        """Get hardware info for a worker"""
        result = await self.db.execute(select(Worker).where(Worker.id == worker_id))
        worker = result.scalar_one_or_none()

        if not worker:
            return {"error": "Worker not found"}

        gpu_info = worker.gpu_info or []

        # Determine the unit divisor from memory_total (which is always large)
        # memory_total for a typical GPU should be 8-80 GB
        def get_divisor(memory_total: int | float) -> float:
            """Determine the divisor to convert memory values to GB.

            We use memory_total to figure out what unit the values are in:
            - If memory_total > 1 billion: values are in bytes
            - If memory_total > 1 million: values are in KB
            - If memory_total > 1000: values are in MB
            - Otherwise: values are already in GB
            """
            if memory_total > 1_000_000_000:
                return 1024 * 1024 * 1024  # bytes to GB
            elif memory_total > 1_000_000:
                return 1024 * 1024  # KB to GB
            elif memory_total > 1000:
                return 1024  # MB to GB
            else:
                return 1  # already GB

        def convert_gpu_memory(gpu: dict) -> dict:
            """Convert a single GPU's memory values to GB."""
            mem_total = gpu.get("memory_total", 0)
            mem_used = gpu.get("memory_used", 0)

            divisor = get_divisor(mem_total)

            return {
                "memory_total_gb": round(mem_total / divisor, 1) if mem_total else 0,
                "memory_used_gb": round(mem_used / divisor, 1) if mem_used else 0,
                "memory_free_gb": round((mem_total - mem_used) / divisor, 1) if mem_total else 0,
            }

        # Convert GPU memory values
        gpus_converted = []
        total_vram_gb = 0
        for i, g in enumerate(gpu_info):
            mem = convert_gpu_memory(g)
            gpus_converted.append(
                {
                    "index": g.get("index", i),
                    "name": g.get("name", "Unknown"),
                    "memory_total_gb": mem["memory_total_gb"],
                    "memory_used_gb": mem["memory_used_gb"],
                    "memory_free_gb": mem["memory_free_gb"],
                    "utilization_percent": g.get("utilization_gpu", 0),
                }
            )
            total_vram_gb += mem["memory_total_gb"]

        return {
            "worker_id": worker.id,
            "worker_name": worker.name,
            "status": worker.status,
            "gpu_count": len(gpu_info),
            "gpus": gpus_converted,
            "total_vram_gb": round(total_vram_gb, 1),
        }

    async def _tool_get_model_info(self, model_id: int) -> dict:
        """Get model info"""
        result = await self.db.execute(select(LLMModel).where(LLMModel.id == model_id))
        model = result.scalar_one_or_none()

        if not model:
            return {"error": "Model not found"}

        # Extract model family from name
        model_family = _extract_model_family(model.name)

        return {
            "model_id": model.id,
            "name": model.name,
            "model_id_hf": model.model_id,
            "source": model.source,
            "model_family": model_family,
            "default_backend": model.backend,
        }

    async def _tool_query_knowledge_base(
        self, model_family: str | None = None, gpu_model: str | None = None
    ) -> dict:
        """Query knowledge base for similar configurations"""
        stmt = select(PerformanceKnowledge)

        if model_family:
            stmt = stmt.where(PerformanceKnowledge.model_family.ilike(f"%{model_family}%"))
        if gpu_model:
            stmt = stmt.where(PerformanceKnowledge.gpu_model.ilike(f"%{gpu_model}%"))

        stmt = stmt.order_by(PerformanceKnowledge.score.desc().nulls_last()).limit(5)

        result = await self.db.execute(stmt)
        records = result.scalars().all()

        if not records:
            return {
                "found": 0,
                "message": "No historical data found. You'll need to run benchmarks to gather data.",
                "records": [],
            }

        return {
            "found": len(records),
            "records": [
                {
                    "model_name": r.model_name,
                    "model_family": r.model_family,
                    "gpu_model": r.gpu_model,
                    "gpu_count": r.gpu_count,
                    "engine": r.engine,
                    "quantization": r.quantization,
                    "tensor_parallel": r.tensor_parallel,
                    "throughput_tps": r.throughput_tps,
                    "ttft_ms": r.ttft_ms,
                    "tpot_ms": r.tpot_ms,
                    "score": r.score,
                }
                for r in records
            ],
        }

    async def _tool_list_deployments(self, worker_id: int) -> dict:
        """List all deployments on a worker"""
        try:
            result = await self.db.execute(
                select(Deployment)
                .where(Deployment.worker_id == worker_id)
                .options(selectinload(Deployment.model))
            )
            deployments = result.scalars().all()

            if not deployments:
                return {
                    "worker_id": worker_id,
                    "count": 0,
                    "deployments": [],
                    "message": "No deployments found on this worker. GPU memory may be used by processes outside LMStack.",
                }

            deployment_list = []
            for d in deployments:
                deployment_list.append(
                    {
                        "deployment_id": d.id,
                        "name": d.name,
                        "model_name": d.model.name if d.model else "Unknown",
                        "status": d.status,
                        "backend": d.backend,
                        "port": d.port,
                        "container_id": d.container_id[:12] if d.container_id else None,
                    }
                )

            return {
                "worker_id": worker_id,
                "count": len(deployments),
                "deployments": deployment_list,
                "message": f"Found {len(deployments)} deployment(s). Stop running deployments to free GPU memory.",
            }
        except Exception as e:
            logger.exception(f"Failed to list deployments: {e}")
            return {"error": str(e)}

    async def _tool_deploy_model(
        self,
        model_id: int,
        worker_id: int,
        engine: str,
        gpu_indexes: list[int] | None = None,
        extra_params: dict | None = None,
    ) -> dict:
        """Deploy a model with specific configuration"""
        from app.services.deployer import DeployerService

        try:
            # Check if there are any pending deployments from this tuning job
            if self.created_deployments:
                return {
                    "success": False,
                    "error": f"You still have active deployments: {self.created_deployments}. "
                    f"Please stop them first using stop_deployment before creating a new one.",
                }

            # Check GPU memory availability
            worker_result = await self.db.execute(select(Worker).where(Worker.id == worker_id))
            worker = worker_result.scalar_one_or_none()
            if worker and worker.gpu_info:
                for g in worker.gpu_info:
                    mem_total = g.get("memory_total", 0)
                    mem_used = g.get("memory_used", 0)
                    # Check if less than 20% memory is free
                    if mem_total > 0 and (mem_total - mem_used) / mem_total < 0.2:
                        free_pct = round((mem_total - mem_used) / mem_total * 100, 1)
                        return {
                            "success": False,
                            "error": f"GPU memory is low (only {free_pct}% free). "
                            f"Please stop any existing deployments first.",
                        }

            # Get model to generate deployment name
            model_result = await self.db.execute(select(LLMModel).where(LLMModel.id == model_id))
            model = model_result.scalar_one_or_none()
            if not model:
                return {"success": False, "error": "Model not found"}

            # Generate unique deployment name
            import time

            deploy_name = f"tuning-{model.name.replace('/', '-')[:30]}-{int(time.time())}"

            # Create deployment
            deployment = Deployment(
                name=deploy_name,
                model_id=model_id,
                worker_id=worker_id,
                backend=engine,
                gpu_indexes=gpu_indexes or [0],
                extra_params=extra_params or {},
                status=DeploymentStatus.PENDING.value,
            )

            self.db.add(deployment)
            await self.db.commit()
            await self.db.refresh(deployment)

            self.created_deployments.append(deployment.id)

            # Start deployment in background using DeployerService
            deployer = DeployerService()
            asyncio.create_task(deployer.deploy(deployment.id))

            return {
                "success": True,
                "deployment_id": deployment.id,
                "deployment_name": deploy_name,
                "config": {
                    "engine": engine,
                    "gpu_indexes": gpu_indexes or [0],
                    "extra_params": extra_params,
                },
                "message": "Deployment created. Use wait_for_deployment to wait until ready.",
            }
        except Exception as e:
            logger.exception(f"Failed to deploy model: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_wait_for_deployment(
        self, deployment_id: int, timeout_seconds: int = 300
    ) -> dict:
        """Wait for deployment to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            result = await self.db.execute(
                select(Deployment)
                .where(Deployment.id == deployment_id)
                .options(selectinload(Deployment.worker))
            )
            deployment = result.scalar_one_or_none()

            if not deployment:
                return {
                    "success": False,
                    "deployment_id": deployment_id,
                    "error": "Deployment not found. It may have been deleted.",
                }

            if deployment.status == DeploymentStatus.RUNNING.value:
                return {
                    "success": True,
                    "deployment_id": deployment_id,
                    "status": "running",
                    "port": deployment.port,
                    "endpoint": f"http://{deployment.worker.address.split(':')[0] if deployment.worker else 'localhost'}:{deployment.port}/v1",
                    "wait_time_seconds": round(time.time() - start_time, 1),
                }
            elif deployment.status in [
                DeploymentStatus.ERROR.value,
                DeploymentStatus.STOPPED.value,
            ]:
                return {
                    "success": False,
                    "deployment_id": deployment_id,
                    "status": deployment.status,
                    "error": deployment.status_message or "Deployment failed",
                    "action_required": "Call stop_deployment to clean up before trying again",
                }

            await asyncio.sleep(5)

        return {
            "success": False,
            "deployment_id": deployment_id,
            "error": f"Timeout after {timeout_seconds}s",
            "action_required": (
                f"1. Call get_deployment_logs({deployment_id}) to check what's happening\n"
                f"2. If model is still loading, wait more with wait_for_deployment(timeout_seconds=300)\n"
                f"3. If there's an error, call stop_deployment({deployment_id}) and try a different config"
            ),
        }

    async def _tool_run_benchmark(
        self,
        deployment_id: int,
        num_requests: int = 20,
        concurrency: int = 4,
        input_tokens: int = 128,
        output_tokens: int = 64,
    ) -> dict:
        """Run actual benchmark against deployment"""
        result = await self.db.execute(
            select(Deployment)
            .where(Deployment.id == deployment_id)
            .options(
                selectinload(Deployment.worker),
                selectinload(Deployment.model),
            )
        )
        deployment = result.scalar_one_or_none()

        if not deployment:
            return {"error": "Deployment not found"}

        if deployment.status != DeploymentStatus.RUNNING.value:
            return {"error": f"Deployment is not running (status: {deployment.status})"}

        # Build endpoint URL
        worker = deployment.worker
        worker_ip = worker.address.split(":")[0]
        base_url = f"http://{worker_ip}:{deployment.port}/v1"

        # Get the model name for API calls
        model_name = deployment.model.model_id if deployment.model else "default"

        # Run benchmark
        metrics = await _run_http_benchmark(
            base_url=base_url,
            model_name=model_name,
            num_requests=num_requests,
            concurrency=concurrency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return metrics

    async def _tool_stop_deployment(self, deployment_id: int) -> dict:
        """Stop and remove a deployment"""
        from app.services.deployer import DeployerService

        try:
            # Get deployment
            result = await self.db.execute(select(Deployment).where(Deployment.id == deployment_id))
            deployment = result.scalar_one_or_none()

            if not deployment:
                return {"success": False, "error": "Deployment not found"}

            # Stop container if running
            if deployment.container_id:
                deployer = DeployerService()
                await deployer.stop(deployment_id)

            # Delete deployment record
            await self.db.delete(deployment)
            await self.db.commit()

            if deployment_id in self.created_deployments:
                self.created_deployments.remove(deployment_id)

            return {"success": True, "message": f"Deployment {deployment_id} stopped and removed"}
        except Exception as e:
            logger.exception(f"Failed to stop deployment: {e}")
            return {"success": False, "error": str(e)}

    async def _tool_check_deployment_status(self, deployment_id: int) -> dict:
        """Check the current status of a deployment without waiting"""
        try:
            result = await self.db.execute(
                select(Deployment)
                .where(Deployment.id == deployment_id)
                .options(selectinload(Deployment.worker))
            )
            deployment = result.scalar_one_or_none()

            if not deployment:
                return {"error": "Deployment not found"}

            return {
                "deployment_id": deployment_id,
                "status": deployment.status,
                "status_message": deployment.status_message,
                "container_id": deployment.container_id,
                "port": deployment.port,
                "backend": deployment.backend,
                "is_ready": deployment.status == DeploymentStatus.RUNNING.value,
                "is_failed": deployment.status == DeploymentStatus.ERROR.value,
                "is_loading": deployment.status == DeploymentStatus.STARTING.value,
            }
        except Exception as e:
            logger.exception(f"Failed to check deployment status: {e}")
            return {"error": str(e)}

    async def _tool_test_deployment_endpoint(self, deployment_id: int) -> dict:
        """Test if the deployment API endpoint is responding"""
        try:
            result = await self.db.execute(
                select(Deployment)
                .where(Deployment.id == deployment_id)
                .options(selectinload(Deployment.worker))
            )
            deployment = result.scalar_one_or_none()

            if not deployment:
                return {"error": "Deployment not found"}

            if not deployment.worker or not deployment.port:
                return {
                    "deployment_id": deployment_id,
                    "ready": False,
                    "error": "Deployment not fully initialized (no worker or port)",
                }

            # Build endpoint URL
            worker = deployment.worker
            worker_ip = worker.address.split(":")[0]
            base_url = f"http://{worker_ip}:{deployment.port}/v1"

            # Test the /v1/models endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    response = await client.get(f"{base_url}/models")
                    if response.status_code == 200:
                        data = response.json()
                        models = data.get("data", [])
                        if models:
                            return {
                                "deployment_id": deployment_id,
                                "ready": True,
                                "endpoint": base_url,
                                "models": [m.get("id") for m in models],
                                "message": "Deployment is ready! You can now run benchmarks.",
                            }
                        else:
                            return {
                                "deployment_id": deployment_id,
                                "ready": False,
                                "endpoint": base_url,
                                "message": "API responding but no models loaded yet",
                            }
                    else:
                        return {
                            "deployment_id": deployment_id,
                            "ready": False,
                            "endpoint": base_url,
                            "status_code": response.status_code,
                            "message": f"API returned status {response.status_code}",
                        }
                except httpx.ConnectError:
                    return {
                        "deployment_id": deployment_id,
                        "ready": False,
                        "endpoint": base_url,
                        "message": "Cannot connect to endpoint - container may still be starting",
                    }
                except httpx.ReadTimeout:
                    return {
                        "deployment_id": deployment_id,
                        "ready": False,
                        "endpoint": base_url,
                        "message": "Connection timeout - model may still be loading",
                    }
        except Exception as e:
            logger.exception(f"Failed to test deployment endpoint: {e}")
            return {"error": str(e)}

    async def _tool_get_deployment_logs(self, deployment_id: int, tail: int = 100) -> dict:
        """Get Docker container logs for a deployment"""
        from app.services.deployer import DeployerService

        try:
            # Get deployment with worker
            result = await self.db.execute(
                select(Deployment)
                .where(Deployment.id == deployment_id)
                .options(selectinload(Deployment.worker))
            )
            deployment = result.scalar_one_or_none()

            if not deployment:
                return {"error": "Deployment not found"}

            if not deployment.container_id:
                return {
                    "deployment_id": deployment_id,
                    "status": deployment.status,
                    "error": "No container ID - deployment may not have started yet",
                    "status_message": deployment.status_message,
                }

            # Use DeployerService to get logs (handles both local and remote)
            deployer = DeployerService()
            logs = await deployer.get_logs(deployment, tail=tail)

            return {
                "deployment_id": deployment_id,
                "container_id": deployment.container_id,
                "status": deployment.status,
                "status_message": deployment.status_message,
                "logs": logs,
            }
        except Exception as e:
            logger.exception(f"Failed to get deployment logs: {e}")
            return {"error": str(e)}

    async def _tool_finish_tuning(
        self, best_config: dict, reasoning: str, all_results: list | None = None
    ) -> dict:
        """Mark tuning as complete and save to knowledge base"""
        # Update job status
        self.job.status = TuningJobStatus.COMPLETED.value
        self.job.status_message = "Auto-tuning completed successfully"
        self.job.best_config = {**best_config, "reasoning": reasoning}
        self.job.all_results = all_results or []
        self.job.completed_at = datetime.now(UTC)

        # Update progress to 100%
        # Use the total_steps from current progress (set during agent loop) or default
        current_total = self.job.progress.get("total_steps", 20) if self.job.progress else 20
        self.job.current_step = current_total
        self.job.total_steps = current_total
        self.job.progress = {
            "step": current_total,
            "total_steps": current_total,
            "step_name": "completed",
            "step_description": "Tuning completed successfully",
            "configs_tested": len(all_results) if all_results else 1,
            "configs_total": len(all_results) if all_results else 1,
        }

        # Save results to knowledge base
        saved_count = 0
        if all_results:
            # Get model and worker info for knowledge base
            model = self.job.model
            worker = self.job.worker
            gpu_info = worker.gpu_info[0] if worker.gpu_info else {}
            gpu_name = gpu_info.get("name", "Unknown GPU")

            for result in all_results:
                metrics = result.get("metrics", {})
                if not metrics:
                    continue

                # Create knowledge record
                knowledge = PerformanceKnowledge(
                    gpu_model=gpu_name,
                    gpu_count=len(result.get("gpu_indexes", [0])),
                    total_vram_gb=sum(
                        (
                            g.get("memory_total", 0) / (1024**3)
                            if g.get("memory_total", 0) > 1_000_000
                            else g.get("memory_total", 0)
                        )
                        for g in (worker.gpu_info or [])
                    ),
                    model_name=model.name,
                    model_family=_extract_model_family(model.name),
                    engine=result.get("engine", best_config.get("engine", "vllm")),
                    quantization=result.get("extra_params", {}).get("quantization"),
                    tensor_parallel=len(result.get("gpu_indexes", [0])),
                    extra_args=result.get("extra_params"),
                    throughput_tps=metrics.get("throughput_tps", 0),
                    ttft_ms=metrics.get("avg_ttft_ms", 0),
                    tpot_ms=metrics.get("avg_tpot_ms", 0),
                    input_length=128,  # Default test params
                    output_length=64,
                    concurrency=4,
                    score=metrics.get("throughput_tps", 0),  # For throughput optimization
                    source_tuning_job_id=self.job.id,
                )
                self.db.add(knowledge)
                saved_count += 1

        await self.db.commit()

        return {
            "success": True,
            "message": f"Tuning completed. Saved {saved_count} result(s) to knowledge base.",
            "best_config": best_config,
            "reasoning": reasoning,
        }

    async def _tool_abort_tuning(self, reason: str) -> dict:
        """Abort the tuning process"""
        self.job.status = TuningJobStatus.FAILED.value
        self.job.status_message = f"Aborted: {reason}"
        self.job.completed_at = datetime.now(UTC)

        # Update progress to show aborted state
        self.job.progress = {
            "step": self.job.current_step,
            "total_steps": self.job.total_steps,
            "step_name": "aborted",
            "step_description": reason,
        }

        await self.db.commit()

        return {"success": True, "message": "Tuning aborted", "reason": reason}

    async def cleanup(self):
        """Clean up any deployments created during tuning"""
        for deployment_id in self.created_deployments:
            try:
                await self._tool_stop_deployment(deployment_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup deployment {deployment_id}: {e}")


# =============================================================================
# Benchmark Implementation
# =============================================================================


async def _run_http_benchmark(
    base_url: str,
    model_name: str = "default",
    num_requests: int = 20,
    concurrency: int = 4,
    input_tokens: int = 128,
    output_tokens: int = 64,
) -> dict:
    """Run actual HTTP benchmark against an OpenAI-compatible endpoint"""

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


def _extract_model_family(model_name: str) -> str:
    """Extract model family from name"""
    name_lower = model_name.lower()
    families = {
        "qwen": "Qwen",
        "llama": "Llama",
        "mistral": "Mistral",
        "deepseek": "DeepSeek",
        "phi": "Phi",
        "gemma": "Gemma",
        "yi": "Yi",
        "glm": "GLM",
    }
    for key, value in families.items():
        if key in name_lower:
            return value
    return "Unknown"


# =============================================================================
# Main Agent Runner
# =============================================================================


async def run_tuning_agent(job_id: int, llm_config: dict | None = None):
    """Run the Auto-Tuning Agent for a job

    Args:
        job_id: The tuning job ID
        llm_config: Optional LLM configuration from chat panel:
            - deployment_id: Use a local deployment
            - base_url: Custom endpoint URL
            - api_key: API key for the endpoint
            - model: Model name
    """
    settings = get_settings()

    async with async_session_maker() as db:
        # Load job with relationships
        result = await db.execute(
            select(TuningJob)
            .where(TuningJob.id == job_id)
            .options(
                selectinload(TuningJob.model),
                selectinload(TuningJob.worker),
            )
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Tuning job {job_id} not found")
            return

        # Initialize tool executor
        executor = AgentToolExecutor(db, job)

        try:
            # Determine LLM configuration (priority: llm_config > settings > auto-detect)
            api_key = None
            base_url = None
            model_name = "gpt-4o"

            if llm_config:
                # Use config from chat panel
                if llm_config.get("deployment_id"):
                    # Use specified local deployment
                    from app.models.deployment import Deployment, DeploymentStatus

                    deploy_result = await db.execute(
                        select(Deployment)
                        .where(Deployment.id == llm_config["deployment_id"])
                        .options(selectinload(Deployment.worker), selectinload(Deployment.model))
                    )
                    deployment = deploy_result.scalar_one_or_none()

                    if deployment and deployment.worker:
                        worker_ip = deployment.worker.address.split(":")[0]
                        base_url = f"http://{worker_ip}:{deployment.port}/v1"
                        api_key = "dummy"
                        model_name = deployment.model.model_id if deployment.model else model_name
                        logger.info(
                            f"Using specified deployment as agent LLM: {base_url} ({model_name})"
                        )
                    else:
                        job.status = TuningJobStatus.FAILED.value
                        job.status_message = (
                            f"Deployment {llm_config['deployment_id']} not found or not running"
                        )
                        await db.commit()
                        return
                elif llm_config.get("base_url"):
                    # Use custom endpoint
                    base_url = llm_config["base_url"]
                    api_key = llm_config.get("api_key") or "dummy"
                    model_name = llm_config.get("model") or model_name
                    logger.info(f"Using custom endpoint as agent LLM: {base_url} ({model_name})")

            # Fall back to settings if no llm_config
            if not api_key:
                api_key = settings.openai_api_key
                base_url = settings.openai_base_url
                model_name = settings.openai_model or model_name

            # If still no API key, try to find any running deployment
            if not api_key:
                from app.models.deployment import Deployment, DeploymentStatus

                deploy_result = await db.execute(
                    select(Deployment)
                    .where(Deployment.status == DeploymentStatus.RUNNING.value)
                    .options(selectinload(Deployment.worker), selectinload(Deployment.model))
                    .limit(1)
                )
                local_deployment = deploy_result.scalar_one_or_none()

                if local_deployment and local_deployment.worker:
                    worker_ip = local_deployment.worker.address.split(":")[0]
                    base_url = f"http://{worker_ip}:{local_deployment.port}/v1"
                    api_key = "dummy"
                    model_name = (
                        local_deployment.model.model_id if local_deployment.model else model_name
                    )
                    logger.info(
                        f"Auto-detected local deployment as agent LLM: {base_url} ({model_name})"
                    )
                else:
                    job.status = TuningJobStatus.FAILED.value
                    job.status_message = (
                        "No LLM configured for Auto-Tuning Agent. "
                        "Please select a model in the chat panel, or deploy a model first."
                    )
                    await db.commit()
                    return

            # Initialize OpenAI client (supports OpenAI-compatible endpoints)
            client = AsyncOpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")

            # Build initial user message
            user_message = f"""Help me find the best deployment configuration for {job.model.name} on {job.worker.name}. I want to optimize for {job.optimization_target}.

Model ID: {job.model_id}, Worker ID: {job.worker_id}"""

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            # Initialize conversation log for UI display
            conversation_log = [
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ]

            # Helper to save conversation log
            async def save_log():
                job.conversation_log = conversation_log
                await db.commit()

            # Update job status
            job.status = TuningJobStatus.ANALYZING.value
            job.status_message = "Agent is analyzing the environment..."
            job.conversation_log = conversation_log
            await db.commit()

            # Agent loop
            max_iterations = 20
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Check if cancelled
                await db.refresh(job)
                if job.status == TuningJobStatus.CANCELLED.value:
                    logger.info(f"Job {job_id} was cancelled")
                    await executor.cleanup()
                    return

                # Call LLM
                logger.info(f"Agent iteration {iteration}, calling LLM with model: {model_name}...")

                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=get_agent_tools(),
                    tool_choice="auto",
                    max_tokens=4096,
                )

                assistant_message = response.choices[0].message
                messages.append(assistant_message.model_dump(exclude_none=True))

                # Add assistant message to conversation log
                log_entry = {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                if assistant_message.tool_calls:
                    log_entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        for tc in assistant_message.tool_calls
                    ]
                conversation_log.append(log_entry)
                await save_log()

                # Check if no tool calls - prompt the agent to take action
                if not assistant_message.tool_calls:
                    logger.warning(f"Agent responded without tool calls at iteration {iteration}")
                    # Add a user message to prompt the agent to take action
                    prompt_message = (
                        "You need to call a tool to proceed. Available actions:\n"
                        "1. list_deployments - Find existing deployments on the worker\n"
                        "2. stop_deployment - Stop a deployment to free GPU memory\n"
                        "3. deploy_model - Deploy a model with specific config\n"
                        "4. test_deployment_endpoint - Check if deployment is ready\n"
                        "5. get_deployment_logs - Check container logs\n"
                        "6. run_benchmark - Run performance benchmark\n"
                        "7. finish_tuning - Complete with recommendation\n"
                        "8. abort_tuning - Abort if cannot proceed\n"
                        "Do not respond with just text - you must call a tool."
                    )
                    messages.append({"role": "user", "content": prompt_message})
                    conversation_log.append(
                        {
                            "role": "user",
                            "content": prompt_message,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    await save_log()
                    continue  # Continue the loop to get tool calls

                # Execute tool calls
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Executing tool: {tool_name}({tool_args})")

                    # Update job progress
                    job.status_message = f"Executing: {tool_name}"
                    job.progress = {
                        "step": iteration,
                        "total_steps": max_iterations,
                        "step_name": tool_name,
                        "step_description": f"Executing {tool_name} with args: {tool_args}",
                        "configs_tested": 0,
                        "configs_total": 0,
                    }
                    await db.commit()

                    # Execute tool
                    result = await executor.execute(tool_name, tool_args)

                    # Add tool result to conversation log
                    conversation_log.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    await save_log()

                    # Check if this was a termination tool
                    if tool_name == "finish_tuning":
                        logger.info(f"Agent completed tuning for job {job_id}")
                        return
                    if tool_name == "abort_tuning":
                        logger.info(f"Agent aborted tuning for job {job_id}")
                        return

                    # Add tool result to messages
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                    )

            # If we reached max iterations without finishing
            job.status = TuningJobStatus.FAILED.value
            job.status_message = "Agent reached maximum iterations without completing"
            await db.commit()

        except Exception as e:
            logger.exception(f"Agent error for job {job_id}: {e}")
            job.status = TuningJobStatus.FAILED.value
            job.status_message = f"Agent error: {str(e)}"
            await db.commit()

        finally:
            # Cleanup any test deployments
            await executor.cleanup()
