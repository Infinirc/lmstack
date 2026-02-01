"""Main agent runner for auto-tuning.

This module contains the main run_tuning_agent function that orchestrates
the auto-tuning process using an LLM-driven agent.
"""

import json
import logging
from datetime import UTC, datetime

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import async_session_maker
from app.models.tuning import TuningJob, TuningJobStatus

from .executor import AgentToolExecutor
from .tools import AGENT_SYSTEM_PROMPT, get_agent_tools

logger = logging.getLogger(__name__)


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
                    from app.models.deployment import Deployment

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

            # Build initial user message with explicit steps
            user_message = f"""Find the optimal deployment configuration for {job.model.name} on {job.worker.name}.
Optimization target: {job.optimization_target}
Model ID: {job.model_id}, Worker ID: {job.worker_id}

REQUIRED STEPS (you must complete all of these):
1. Call get_hardware_info(worker_id={job.worker_id}) to check GPU specs
2. Call query_knowledge_base() to check historical data
3. Deploy the model with deploy_model() and wait for it
4. Run run_benchmark() to test performance
5. Stop the deployment and optionally test other configurations
6. Call finish_tuning() with best_config and all benchmark results

Start with Step 1: get_hardware_info"""

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

            # Agent loop - limit iterations to prevent infinite loops
            max_iterations = 15
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

                # Force tool calls if essential steps not completed
                # Use "required" to ensure tool is called when needed
                if not executor.hardware_checked or (
                    not executor.benchmark_results and iteration < 10
                ):
                    tool_choice = "required"
                else:
                    tool_choice = "auto"

                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=get_agent_tools(),
                    tool_choice=tool_choice,
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
                    # Build a context-aware prompt based on current state
                    if not executor.hardware_checked:
                        prompt_message = (
                            f"You must call get_hardware_info(worker_id={job.worker_id}) first "
                            "to check the GPU environment before proceeding."
                        )
                    elif not executor.benchmark_results:
                        prompt_message = (
                            "You must run at least one benchmark before finishing. "
                            f"Call deploy_model(model_id={job.model_id}, worker_id={job.worker_id}, engine='vllm') "
                            "to deploy the model, then run run_benchmark() after it's ready."
                        )
                    else:
                        prompt_message = (
                            "You have benchmark results. Call finish_tuning() with the best configuration "
                            "to complete the tuning process."
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
