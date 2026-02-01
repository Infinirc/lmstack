"""AgentToolExecutor - Executes agent tools with real system interactions.

This module contains the AgentToolExecutor class that handles executing
tools called by the auto-tuning LLM agent.
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.deployment import Deployment, DeploymentStatus
from app.models.llm_model import LLMModel
from app.models.tuning import PerformanceKnowledge, TuningJob, TuningJobStatus
from app.models.worker import Worker

from .benchmark import run_http_benchmark
from .helpers import extract_model_family

logger = logging.getLogger(__name__)


class AgentToolExecutor:
    """Execute agent tools with real system interactions"""

    def __init__(self, db: AsyncSession, job: TuningJob):
        self.db = db
        self.job = job
        self.created_deployments: list[int] = []
        self.benchmark_results: list[dict] = []  # Track completed benchmarks
        self.hardware_checked: bool = False  # Track if hardware was checked

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
        self.hardware_checked = True  # Mark hardware as checked
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
        model_family = extract_model_family(model.name)

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

            # Update job status to show deployment progress
            elapsed = int(time.time() - start_time)
            status_map = {
                "pending": "Preparing deployment...",
                "starting": "Loading model into GPU memory...",
                "running": "Model ready!",
                "error": "Deployment failed",
                "stopped": "Deployment stopped",
            }
            status_desc = status_map.get(deployment.status, deployment.status)
            self.job.status_message = f"Waiting for model: {status_desc} ({elapsed}s)"
            self.job.progress = {
                "step": self.job.progress.get("step", 0) if self.job.progress else 0,
                "total_steps": (
                    self.job.progress.get("total_steps", 15) if self.job.progress else 15
                ),
                "step_name": "wait_for_deployment",
                "step_description": f"Deployment #{deployment_id}: {status_desc}",
                "deployment_status": deployment.status,
                "deployment_message": deployment.status_message or "",
                "elapsed_seconds": elapsed,
            }
            await self.db.commit()

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
                f"1. Call test_deployment_endpoint({deployment_id}) to check if it's actually ready\n"
                f"2. If not ready, call stop_deployment({deployment_id}) and try next config\n"
                f"DO NOT wait again - move on to the next configuration!"
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
        metrics = await run_http_benchmark(
            base_url=base_url,
            model_name=model_name,
            num_requests=num_requests,
            concurrency=concurrency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Save successful benchmark results for tracking
        if metrics.get("success"):
            self.benchmark_results.append(
                {
                    "deployment_id": deployment_id,
                    "engine": deployment.backend,
                    "gpu_indexes": deployment.gpu_indexes or [0],
                    "extra_params": deployment.extra_params or {},
                    "metrics": metrics.get("metrics", {}),
                }
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
                try:
                    deployer = DeployerService()
                    await deployer.stop(deployment_id)
                except Exception as e:
                    logger.warning(f"Failed to stop container for deployment {deployment_id}: {e}")

            # Always update status to stopped first (in case delete fails)
            deployment.status = DeploymentStatus.STOPPED.value
            deployment.status_message = "Stopped by tuning agent"
            await self.db.commit()

            # Then try to delete
            try:
                await self.db.delete(deployment)
                await self.db.commit()
            except Exception as e:
                logger.warning(f"Failed to delete deployment {deployment_id}: {e}")
                # Status is already stopped, so it's ok

            if deployment_id in self.created_deployments:
                self.created_deployments.remove(deployment_id)

            return {"success": True, "message": f"Deployment {deployment_id} stopped and removed"}
        except Exception as e:
            logger.exception(f"Failed to stop deployment: {e}")
            # Try to at least mark it stopped
            try:
                result = await self.db.execute(
                    select(Deployment).where(Deployment.id == deployment_id)
                )
                deployment = result.scalar_one_or_none()
                if deployment:
                    deployment.status = DeploymentStatus.STOPPED.value
                    await self.db.commit()
            except Exception:
                pass
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
        # Validate that proper steps were completed
        if not self.hardware_checked:
            return {
                "success": False,
                "error": "Cannot finish tuning: You must call get_hardware_info first to check the GPU environment.",
                "required_action": "Call get_hardware_info(worker_id=...) before finishing.",
            }

        if not self.benchmark_results and not all_results:
            return {
                "success": False,
                "error": "Cannot finish tuning: No benchmark results found. You must run at least one benchmark.",
                "required_action": "Deploy a model, run run_benchmark(), then call finish_tuning with the results.",
            }

        # Use tracked benchmark results if all_results not provided
        if not all_results:
            all_results = self.benchmark_results

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
                    model_family=extract_model_family(model.name),
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
