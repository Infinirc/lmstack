/**
 * Benchmark Tools for Auto-Tuning
 *
 * Provides tools for running performance benchmarks on LLM deployments.
 */

import { LMStackClient } from "../client.js";

export interface BenchmarkConfig {
  deploymentId: number;
  durationSeconds?: number;
  inputLength?: number;
  outputLength?: number;
  concurrency?: number;
}

export interface BenchmarkResult {
  success: boolean;
  throughputTps?: number;
  ttftMs?: number;
  tpotMs?: number;
  totalLatencyMs?: number;
  gpuUtilization?: number;
  vramUsageGb?: number;
  error?: string;
  rawResults?: Record<string, unknown>;
}

/**
 * Run a throughput benchmark on a deployment
 */
export async function runBenchmark(
  client: LMStackClient,
  config: BenchmarkConfig
): Promise<string> {
  const {
    deploymentId,
    durationSeconds = 60,
    inputLength = 512,
    outputLength = 128,
    concurrency = 1,
  } = config;

  try {
    // First, get the deployment info
    const deployment = await client.getDeployment(deploymentId);
    if (!deployment) {
      return `## Benchmark Failed\n\nDeployment ID ${deploymentId} not found.`;
    }

    if (deployment.status !== "running") {
      return `## Benchmark Failed\n\nDeployment "${deployment.name}" is not running (status: ${deployment.status}).`;
    }

    // Run the benchmark via API
    const result = await client.runBenchmark({
      deployment_id: deploymentId,
      test_type: "throughput",
      duration_seconds: durationSeconds,
      input_length: inputLength,
      output_length: outputLength,
      concurrency: concurrency,
    });

    if (!result || result.error) {
      return `## Benchmark Failed\n\nError: ${result?.error || "Unknown error"}`;
    }

    // Format results
    let output = `## Benchmark Results\n\n`;
    output += `**Deployment:** ${deployment.name}\n`;
    output += `**Model:** ${deployment.model?.name || "Unknown"}\n`;
    output += `**Engine:** ${deployment.engine || "vllm"}\n\n`;

    output += `### Test Configuration\n`;
    output += `- Duration: ${durationSeconds}s\n`;
    output += `- Input Length: ${inputLength} tokens\n`;
    output += `- Output Length: ${outputLength} tokens\n`;
    output += `- Concurrency: ${concurrency}\n\n`;

    output += `### Performance Metrics\n`;
    if (result.throughput_tps !== undefined) {
      output += `- **Throughput:** ${result.throughput_tps.toFixed(2)} tokens/sec\n`;
    }
    if (result.ttft_ms !== undefined) {
      output += `- **TTFT (Time to First Token):** ${result.ttft_ms.toFixed(2)} ms\n`;
    }
    if (result.tpot_ms !== undefined) {
      output += `- **TPOT (Time per Output Token):** ${result.tpot_ms.toFixed(2)} ms\n`;
    }
    if (result.total_latency_ms !== undefined) {
      output += `- **Total Latency:** ${result.total_latency_ms.toFixed(2)} ms\n`;
    }

    output += `\n### Resource Usage\n`;
    if (result.gpu_utilization !== undefined) {
      output += `- **GPU Utilization:** ${result.gpu_utilization.toFixed(1)}%\n`;
    }
    if (result.vram_usage_gb !== undefined) {
      output += `- **VRAM Usage:** ${result.vram_usage_gb.toFixed(2)} GB\n`;
    }

    return output;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    return `## Benchmark Failed\n\nError: ${errMsg}`;
  }
}

/**
 * Start an Auto-Tuning job
 */
export async function startAutoTuning(
  client: LMStackClient,
  modelId: number,
  workerId: number,
  config: {
    engines?: string[];
    tensorParallelSizes?: number[];
    gpuMemoryUtilizations?: number[];
    maxModelLengths?: number[];
    concurrencyLevels?: number[];
    durationSeconds?: number;
    llmBaseUrl?: string;
    llmApiKey?: string;
    llmModel?: string;
  }
): Promise<string> {
  try {
    const tuningConfig = {
      engines: config.engines || ["vllm"],
      parameters: {
        tensor_parallel_size: config.tensorParallelSizes || [1],
        gpu_memory_utilization: config.gpuMemoryUtilizations || [0.85, 0.90],
        max_model_len: config.maxModelLengths || [4096],
      },
      benchmark: {
        duration_seconds: config.durationSeconds || 60,
        input_length: 512,
        output_length: 128,
        concurrency: config.concurrencyLevels || [1, 4],
      },
    };

    // Build LLM config - use provided values or fall back to environment variables
    const llmBaseUrl = config.llmBaseUrl || process.env.AGENT_LLM_BASE_URL;
    const llmApiKey = config.llmApiKey || process.env.AGENT_LLM_API_KEY;
    const llmModel = config.llmModel || process.env.AGENT_LLM_MODEL;

    const llmConfig = llmBaseUrl ? {
      base_url: llmBaseUrl,
      api_key: llmApiKey,
      model: llmModel,
    } : undefined;

    const result = await client.createTuningJob({
      model_id: modelId,
      worker_id: workerId,
      optimization_target: "throughput",
      tuning_config: tuningConfig,
      llm_config: llmConfig,
    });

    if (!result || result.error) {
      return `## Auto-Tuning Failed to Start\n\nError: ${result?.error || "Unknown error"}`;
    }

    // Calculate total configs to test
    const totalConfigs =
      tuningConfig.engines.length *
      tuningConfig.parameters.tensor_parallel_size.length *
      tuningConfig.parameters.gpu_memory_utilization.length *
      tuningConfig.parameters.max_model_len.length *
      tuningConfig.benchmark.concurrency.length;

    let output = `## Auto-Tuning Job Started\n\n`;
    output += `**Job ID:** ${result.id}\n`;
    output += `**Status:** ${result.status}\n\n`;

    output += `### Configuration\n`;
    output += `- Engines: ${tuningConfig.engines.join(", ")}\n`;
    output += `- Tensor Parallel Sizes: ${tuningConfig.parameters.tensor_parallel_size.join(", ")}\n`;
    output += `- GPU Memory Utilizations: ${tuningConfig.parameters.gpu_memory_utilization.join(", ")}\n`;
    output += `- Max Model Lengths: ${tuningConfig.parameters.max_model_len.join(", ")}\n`;
    output += `- Concurrency Levels: ${tuningConfig.benchmark.concurrency.join(", ")}\n\n`;

    output += `**Total configurations to test:** ${totalConfigs}\n\n`;
    output += `---\n`;
    output += `**NEXT STEP:** Call \`wait_for_tuning_job(job_id=${result.id})\` to wait for completion.\n`;
    output += `Do NOT repeatedly call get_tuning_job_status - model loading takes time.`;

    return output;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    return `## Auto-Tuning Failed to Start\n\nError: ${errMsg}`;
  }
}

/**
 * Get Auto-Tuning job status
 */
export async function getTuningJobStatus(
  client: LMStackClient,
  jobId: number
): Promise<string> {
  try {
    const job = await client.getTuningJob(jobId);
    if (!job) {
      return `## Tuning Job Not Found\n\nJob ID ${jobId} not found.`;
    }

    let output = `## Auto-Tuning Job Status\n\n`;
    output += `**Job ID:** ${job.id}\n`;
    output += `**Model:** ${job.model_name || "Unknown"}\n`;
    output += `**Worker:** ${job.worker_name || "Unknown"}\n`;
    output += `**Status:** ${job.status.toUpperCase()}\n`;

    if (job.status_message) {
      output += `**Message:** ${job.status_message}\n`;
    }

    output += `\n### Progress\n`;
    output += `- Step: ${job.current_step}/${job.total_steps}\n`;

    if (job.progress) {
      output += `- Current Step: ${job.progress.step_name || "N/A"}\n`;
      if (job.progress.configs_total > 0) {
        output += `- Configs Tested: ${job.progress.configs_tested}/${job.progress.configs_total}\n`;
      }
      if (job.progress.best_score_so_far != null) {
        output += `- Best Score So Far: ${job.progress.best_score_so_far.toFixed(2)}\n`;
      }
    }

    if (job.status === "completed" && job.best_config) {
      output += `\n### Best Configuration\n`;
      output += "```json\n";
      output += JSON.stringify(job.best_config, null, 2);
      output += "\n```\n";
    }

    if (job.all_results && job.all_results.length > 0) {
      output += `\n### Top Results\n`;
      const topResults = job.all_results.slice(0, 5);
      for (let i = 0; i < topResults.length; i++) {
        const r = topResults[i];
        output += `${i + 1}. Engine: ${r.engine}, TPS: ${r.throughput_tps?.toFixed(2) || "N/A"}\n`;
      }
    }

    // Add strong suggestion for running jobs
    if (!["completed", "failed", "cancelled"].includes(job.status)) {
      output += `\n---\n`;
      output += `**ACTION REQUIRED:** Call \`wait_for_tuning_job(job_id=${jobId})\` to wait for completion.\n`;
      output += `Do NOT call get_tuning_job_status again - it wastes API calls. Model loading takes several minutes.`;
    }

    return output;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    return `## Failed to Get Job Status\n\nError: ${errMsg}`;
  }
}

/**
 * Wait for Auto-Tuning job to complete or reach a terminal state
 */
export async function waitForTuningJob(
  client: LMStackClient,
  jobId: number,
  timeoutSeconds: number = 600,
  pollIntervalSeconds: number = 10
): Promise<string> {
  try {
    const startTime = Date.now();
    const timeoutMs = timeoutSeconds * 1000;
    let lastStatus = "";

    while (Date.now() - startTime < timeoutMs) {
      const job = await client.getTuningJob(jobId);
      if (!job) {
        return `## Tuning Job Not Found\n\nJob ID ${jobId} not found.`;
      }

      // Check for terminal states
      if (job.status === "completed") {
        let output = `## Auto-Tuning Completed Successfully\n\n`;
        output += `**Job ID:** ${job.id}\n`;
        output += `**Model:** ${job.model_name || "Unknown"}\n`;
        output += `**Duration:** ${Math.round((Date.now() - startTime) / 1000)}s\n\n`;

        if (job.best_config) {
          output += `### Best Configuration\n`;
          output += "```json\n";
          output += JSON.stringify(job.best_config, null, 2);
          output += "\n```\n";
        }

        if (job.all_results && job.all_results.length > 0) {
          output += `\n### All Results (${job.all_results.length} configs tested)\n`;
          for (let i = 0; i < job.all_results.length; i++) {
            const r = job.all_results[i];
            output += `${i + 1}. Engine: ${r.engine}, TPS: ${r.throughput_tps?.toFixed(2) || "N/A"}, TTFT: ${r.ttft_ms?.toFixed(0) || "N/A"}ms\n`;
          }
        }

        return output;
      }

      if (job.status === "failed") {
        return `## Auto-Tuning Failed\n\n**Job ID:** ${job.id}\n**Error:** ${job.status_message || "Unknown error"}`;
      }

      if (job.status === "cancelled") {
        return `## Auto-Tuning Cancelled\n\n**Job ID:** ${job.id}\n**Reason:** ${job.status_message || "Cancelled by user"}`;
      }

      // Log progress if status changed
      if (job.status !== lastStatus) {
        lastStatus = job.status;
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollIntervalSeconds * 1000));
    }

    // Timeout reached
    const job = await client.getTuningJob(jobId);
    return `## Wait Timeout\n\nTuning job ${jobId} did not complete within ${timeoutSeconds}s.\n\n**Current Status:** ${job?.status || "Unknown"}\n**Message:** ${job?.status_message || "N/A"}\n\nThe job is still running in the background. Use \`get_tuning_job_status\` to check progress later.`;

  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    return `## Wait Failed\n\nError: ${errMsg}`;
  }
}

/**
 * Compare benchmark results between configurations
 */
export async function compareBenchmarkResults(
  results: Array<{
    name: string;
    throughputTps: number;
    ttftMs: number;
    tpotMs: number;
  }>
): Promise<string> {
  if (results.length === 0) {
    return "## No Results to Compare\n\nProvide at least one result to compare.";
  }

  // Sort by throughput descending
  const sorted = [...results].sort((a, b) => b.throughputTps - a.throughputTps);

  let output = `## Benchmark Comparison\n\n`;
  output += `| Rank | Configuration | Throughput (TPS) | TTFT (ms) | TPOT (ms) |\n`;
  output += `|------|---------------|------------------|-----------|------------|\n`;

  for (let i = 0; i < sorted.length; i++) {
    const r = sorted[i];
    const winner = i === 0 ? " 🏆" : "";
    output += `| ${i + 1} | ${r.name}${winner} | ${r.throughputTps.toFixed(2)} | ${r.ttftMs.toFixed(2)} | ${r.tpotMs.toFixed(2)} |\n`;
  }

  if (sorted.length > 1) {
    const best = sorted[0];
    const worst = sorted[sorted.length - 1];
    const improvement = ((best.throughputTps - worst.throughputTps) / worst.throughputTps * 100).toFixed(1);
    output += `\n**Best configuration is ${improvement}% faster than worst.**`;
  }

  return output;
}
