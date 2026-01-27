#!/usr/bin/env node
/**
 * LMStack MCP Server
 *
 * Model Context Protocol server for LMStack platform.
 * Provides resources and tools for managing LLM infrastructure.
 *
 * This MCP Server exposes the SAME tools as the Web Chat interface,
 * allowing AI agents (e.g., Claude Desktop, Cursor) to manage LMStack.
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { LMStackClient } from "./client.js";
import {
  formatWorkers,
  formatContainers,
  formatDeployments,
  formatModels,
  formatSystemStatus,
  formatApiKeys,
  formatImages,
  formatStorageVolumes,
  formatDiskUsage,
} from "./formatters.js";
import { webSearch, searchLLMConfig } from "./tools/webSearch.js";
import {
  runBenchmark,
  startAutoTuning,
  getTuningJobStatus,
  waitForTuningJob,
} from "./tools/benchmark.js";

// Configuration from environment
const LMSTACK_API_URL = process.env.LMSTACK_API_URL || "http://localhost:8000/api";
const LMSTACK_API_TOKEN = process.env.LMSTACK_API_TOKEN || "";

// Initialize LMStack client
const client = new LMStackClient(LMSTACK_API_URL, LMSTACK_API_TOKEN);

// Create MCP server
const server = new Server(
  {
    name: "lmstack-mcp-server",
    version: "0.2.0",
  },
  {
    capabilities: {
      resources: {},
      tools: {},
    },
  }
);

/**
 * List available resources
 */
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "lmstack://system/status",
        name: "System Status",
        description: "Complete LMStack system status including workers, deployments, and containers",
        mimeType: "text/plain",
      },
      {
        uri: "lmstack://workers",
        name: "Workers",
        description: "List of all worker nodes with GPU information",
        mimeType: "text/plain",
      },
      {
        uri: "lmstack://containers",
        name: "Docker Containers",
        description: "List of all Docker containers across workers",
        mimeType: "text/plain",
      },
      {
        uri: "lmstack://deployments",
        name: "Model Deployments",
        description: "List of all model deployments",
        mimeType: "text/plain",
      },
      {
        uri: "lmstack://models",
        name: "Available Models",
        description: "List of all registered models",
        mimeType: "text/plain",
      },
      {
        uri: "lmstack://api-keys",
        name: "API Keys",
        description: "List of all API keys",
        mimeType: "text/plain",
      },
      {
        uri: "lmstack://images",
        name: "Docker Images",
        description: "List of all Docker images across workers",
        mimeType: "text/plain",
      },
      {
        uri: "lmstack://storage",
        name: "Storage Volumes",
        description: "List of all storage volumes and disk usage",
        mimeType: "text/plain",
      },
    ],
  };
});

/**
 * Read resource content
 */
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  try {
    switch (uri) {
      case "lmstack://system/status": {
        const [workers, containers, deployments, models] = await Promise.all([
          client.getWorkers(),
          client.getContainers(),
          client.getDeployments(),
          client.getModels(),
        ]);
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatSystemStatus(workers, containers, deployments, models),
            },
          ],
        };
      }

      case "lmstack://workers": {
        const workers = await client.getWorkers();
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatWorkers(workers),
            },
          ],
        };
      }

      case "lmstack://containers": {
        const containers = await client.getContainers();
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatContainers(containers),
            },
          ],
        };
      }

      case "lmstack://deployments": {
        const deployments = await client.getDeployments();
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatDeployments(deployments),
            },
          ],
        };
      }

      case "lmstack://models": {
        const models = await client.getModels();
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatModels(models),
            },
          ],
        };
      }

      case "lmstack://api-keys": {
        const apiKeysData = await client.getApiKeys();
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatApiKeys(apiKeysData),
            },
          ],
        };
      }

      case "lmstack://images": {
        const images = await client.getImages();
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatImages(images),
            },
          ],
        };
      }

      case "lmstack://storage": {
        const [volumes, diskUsage] = await Promise.all([
          client.getStorageVolumes(),
          client.getDiskUsage(),
        ]);
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: formatStorageVolumes(volumes) + "\n\n" + formatDiskUsage(diskUsage),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown resource: ${uri}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      contents: [
        {
          uri,
          mimeType: "text/plain",
          text: `Error fetching resource: ${message}`,
        },
      ],
    };
  }
});

/**
 * List available tools
 * These tools match the Web Chat tools in tools.ts
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      // ============== Query Tools ==============
      {
        name: "get_system_status",
        description: "Get complete LMStack system status including workers, GPUs, containers, and deployments",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "list_workers",
        description: "List all worker nodes with their GPU status and memory usage",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "list_containers",
        description: "List all Docker containers running on workers",
        inputSchema: {
          type: "object",
          properties: {
            status: {
              type: "string",
              description: "Filter by status: running, stopped, all (default: all)",
              enum: ["running", "stopped", "all"],
            },
            worker_id: {
              type: "number",
              description: "Filter by specific worker ID",
            },
          },
          required: [],
        },
      },
      {
        name: "list_deployments",
        description: "List all model deployments",
        inputSchema: {
          type: "object",
          properties: {
            status: {
              type: "string",
              description: "Filter by status: running, stopped, all (default: all)",
              enum: ["running", "stopped", "all"],
            },
          },
          required: [],
        },
      },
      {
        name: "list_models",
        description: "List all available models that can be deployed",
        inputSchema: {
          type: "object",
          properties: {
            source: {
              type: "string",
              description: "Filter by source: huggingface, ollama, local (optional)",
              enum: ["huggingface", "ollama", "local"],
            },
          },
          required: [],
        },
      },
      {
        name: "get_gpu_status",
        description: "Get detailed GPU status including memory usage and utilization",
        inputSchema: {
          type: "object",
          properties: {
            worker_id: {
              type: "number",
              description: "Filter by specific worker ID",
            },
          },
          required: [],
        },
      },

      // ============== Model Management Tools ==============
      {
        name: "add_model",
        description: "Add a new model to the system. Supports HuggingFace and Ollama models.",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Model name/identifier (e.g., 'Qwen/Qwen2.5-7B-Instruct' for HuggingFace, 'llama3.2' for Ollama)",
            },
            source: {
              type: "string",
              description: "Model source",
              enum: ["huggingface", "ollama"],
            },
            parameters: {
              type: "string",
              description: "Optional: Model parameters (e.g., '7B', '13B')",
            },
            quantization: {
              type: "string",
              description: "Optional: Quantization format (e.g., 'GPTQ', 'AWQ', 'GGUF')",
            },
          },
          required: ["name", "source"],
        },
      },
      {
        name: "delete_model",
        description: "Delete a model from the system. This will NOT delete any deployments using this model.",
        inputSchema: {
          type: "object",
          properties: {
            model_id: {
              type: "number",
              description: "ID of the model to delete (use list_models to find IDs)",
            },
          },
          required: ["model_id"],
        },
      },

      // ============== Deployment Tools ==============
      {
        name: "deploy_model",
        description: "Deploy a model to a worker. Returns deployment ID on success.",
        inputSchema: {
          type: "object",
          properties: {
            model_id: {
              type: "number",
              description: "ID of the model to deploy",
            },
            worker_id: {
              type: "number",
              description: "ID of the worker to deploy to",
            },
            name: {
              type: "string",
              description: "Name for the deployment (optional, auto-generated if not provided)",
            },
            gpu_indexes: {
              type: "array",
              items: { type: "number" },
              description: "GPU indices to use (optional, defaults to [0])",
            },
            backend: {
              type: "string",
              enum: ["vllm", "sglang", "ollama"],
              description: "Inference backend (optional, defaults to vllm)",
            },
          },
          required: ["model_id", "worker_id"],
        },
      },
      {
        name: "stop_deployment",
        description: "Stop a running model deployment",
        inputSchema: {
          type: "object",
          properties: {
            deployment_id: {
              type: "number",
              description: "ID of the deployment to stop",
            },
          },
          required: ["deployment_id"],
        },
      },
      {
        name: "start_deployment",
        description: "Start a stopped model deployment",
        inputSchema: {
          type: "object",
          properties: {
            deployment_id: {
              type: "number",
              description: "ID of the deployment to start",
            },
          },
          required: ["deployment_id"],
        },
      },
      {
        name: "delete_deployment",
        description: "Delete a model deployment completely. This cannot be undone.",
        inputSchema: {
          type: "object",
          properties: {
            deployment_id: {
              type: "number",
              description: "ID of the deployment to delete",
            },
          },
          required: ["deployment_id"],
        },
      },

      // ============== Container Tools ==============
      {
        name: "stop_container",
        description: "Stop a running Docker container. Use list_containers first to find worker_id.",
        inputSchema: {
          type: "object",
          properties: {
            container_name: {
              type: "string",
              description: "Name of the container to stop",
            },
            worker_id: {
              type: "number",
              description: "ID of the worker where the container is running",
            },
          },
          required: ["container_name", "worker_id"],
        },
      },
      {
        name: "remove_container",
        description: "Remove/delete a Docker container. Use list_containers first to find worker_id.",
        inputSchema: {
          type: "object",
          properties: {
            container_name: {
              type: "string",
              description: "Name of the container to remove",
            },
            worker_id: {
              type: "number",
              description: "ID of the worker where the container is located",
            },
            force: {
              type: "boolean",
              description: "Force remove even if running (default: false)",
            },
          },
          required: ["container_name", "worker_id"],
        },
      },

      // ============== API Key Tools ==============
      {
        name: "list_api_keys",
        description: "List all API keys in the system with their usage statistics.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "create_api_key",
        description: "Create a new API key for accessing the LMStack API.",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Name for the API key (e.g., 'production-key', 'test-key')",
            },
            description: {
              type: "string",
              description: "Optional description for the API key",
            },
            expires_in_days: {
              type: "number",
              description: "Optional: Number of days until the key expires. If not set, the key never expires.",
            },
          },
          required: ["name"],
        },
      },
      {
        name: "delete_api_key",
        description: "Delete an API key from the system.",
        inputSchema: {
          type: "object",
          properties: {
            api_key_id: {
              type: "number",
              description: "ID of the API key to delete (use list_api_keys to find IDs)",
            },
          },
          required: ["api_key_id"],
        },
      },

      // ============== Docker Image Tools ==============
      {
        name: "list_images",
        description: "List all Docker images across all workers.",
        inputSchema: {
          type: "object",
          properties: {
            worker_id: {
              type: "number",
              description: "Optional: Filter by specific worker ID",
            },
            repository: {
              type: "string",
              description: "Optional: Filter by repository name",
            },
          },
          required: [],
        },
      },
      {
        name: "pull_image",
        description: "Pull a Docker image from a registry to a worker.",
        inputSchema: {
          type: "object",
          properties: {
            worker_id: {
              type: "number",
              description: "ID of the worker to pull the image to",
            },
            image: {
              type: "string",
              description: "Image reference (e.g., 'nginx:latest', 'python:3.11')",
            },
          },
          required: ["worker_id", "image"],
        },
      },
      {
        name: "delete_image",
        description: "Delete a Docker image from a worker.",
        inputSchema: {
          type: "object",
          properties: {
            image_id: {
              type: "string",
              description: "ID or name of the image to delete",
            },
            worker_id: {
              type: "number",
              description: "ID of the worker where the image is located",
            },
            force: {
              type: "boolean",
              description: "Force removal even if image is in use (default: false)",
            },
          },
          required: ["image_id", "worker_id"],
        },
      },

      // ============== Storage Tools ==============
      {
        name: "list_storage_volumes",
        description: "List all Docker storage volumes across all workers.",
        inputSchema: {
          type: "object",
          properties: {
            worker_id: {
              type: "number",
              description: "Optional: Filter by specific worker ID",
            },
          },
          required: [],
        },
      },
      {
        name: "get_disk_usage",
        description: "Get Docker disk usage statistics including images, containers, volumes, and build cache.",
        inputSchema: {
          type: "object",
          properties: {
            worker_id: {
              type: "number",
              description: "Optional: Filter by specific worker ID",
            },
          },
          required: [],
        },
      },
      {
        name: "delete_storage_volume",
        description: "Delete a Docker storage volume from a worker.",
        inputSchema: {
          type: "object",
          properties: {
            volume_name: {
              type: "string",
              description: "Name of the volume to delete",
            },
            worker_id: {
              type: "number",
              description: "ID of the worker where the volume is located",
            },
            force: {
              type: "boolean",
              description: "Force removal (default: false)",
            },
          },
          required: ["volume_name", "worker_id"],
        },
      },
      {
        name: "prune_storage",
        description: "Clean up unused Docker resources (images, containers, volumes, build cache) to free disk space.",
        inputSchema: {
          type: "object",
          properties: {
            worker_id: {
              type: "number",
              description: "Optional: Only prune on specific worker. If not set, prunes on all workers.",
            },
            images: {
              type: "boolean",
              description: "Prune unused images (default: true)",
            },
            containers: {
              type: "boolean",
              description: "Prune stopped containers (default: true)",
            },
            volumes: {
              type: "boolean",
              description: "Prune unused volumes (default: false - be careful!)",
            },
            build_cache: {
              type: "boolean",
              description: "Prune build cache (default: true)",
            },
          },
          required: [],
        },
      },

      // ============== Web Search Tools ==============
      {
        name: "web_search",
        description:
          "Search the web for information about LLM deployment configurations, performance benchmarks, and optimization guides. Useful for finding optimal settings for specific GPU + model combinations.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description:
                "Search query (e.g., 'Qwen2.5-7B RTX 4090 vLLM optimal settings')",
            },
            max_results: {
              type: "number",
              description: "Maximum number of results to return (default: 5)",
            },
          },
          required: ["query"],
        },
      },
      {
        name: "search_llm_config",
        description:
          "Search for optimal LLM deployment configurations for a specific GPU and model combination. Returns deployment guides and performance benchmarks.",
        inputSchema: {
          type: "object",
          properties: {
            gpu_model: {
              type: "string",
              description: "GPU model name (e.g., 'RTX 4090', 'A100', 'H100')",
            },
            llm_model: {
              type: "string",
              description:
                "LLM model name (e.g., 'Qwen2.5-7B', 'Llama-3.1-8B', 'Mistral-7B')",
            },
            backend: {
              type: "string",
              description: "Inference backend (default: 'vLLM')",
              enum: ["vLLM", "SGLang", "Ollama", "TGI"],
            },
          },
          required: ["gpu_model", "llm_model"],
        },
      },

      // ============== Auto-Tuning & Benchmark Tools ==============
      {
        name: "run_benchmark",
        description:
          "Run a throughput benchmark on a deployment to measure performance (tokens/sec, latency).",
        inputSchema: {
          type: "object",
          properties: {
            deployment_id: {
              type: "number",
              description: "ID of the deployment to benchmark",
            },
            duration_seconds: {
              type: "number",
              description: "Test duration in seconds (default: 60)",
            },
            input_length: {
              type: "number",
              description: "Input token length (default: 512)",
            },
            output_length: {
              type: "number",
              description: "Output token length (default: 128)",
            },
            concurrency: {
              type: "number",
              description: "Number of concurrent requests (default: 1)",
            },
          },
          required: ["deployment_id"],
        },
      },
      {
        name: "start_auto_tuning",
        description:
          "Start an Auto-Tuning job to find optimal deployment configuration for a model. Tests multiple frameworks and parameter combinations. IMPORTANT: After starting, use wait_for_tuning_job to wait for completion - do NOT repeatedly call get_tuning_job_status.",
        inputSchema: {
          type: "object",
          properties: {
            model_id: {
              type: "number",
              description: "ID of the model to tune",
            },
            worker_id: {
              type: "number",
              description: "ID of the worker to use for testing",
            },
            engines: {
              type: "array",
              items: { type: "string" },
              description: "Inference engines to test (default: ['vllm'])",
            },
            tensor_parallel_sizes: {
              type: "array",
              items: { type: "number" },
              description: "Tensor parallel sizes to test (default: [1])",
            },
            gpu_memory_utilizations: {
              type: "array",
              items: { type: "number" },
              description: "GPU memory utilization values to test (default: [0.85, 0.90])",
            },
            max_model_lengths: {
              type: "array",
              items: { type: "number" },
              description: "Max model lengths to test (default: [4096])",
            },
            concurrency_levels: {
              type: "array",
              items: { type: "number" },
              description: "Concurrency levels to test (default: [1, 4])",
            },
            llm_base_url: {
              type: "string",
              description: "OpenAI-compatible API base URL for the agent LLM (e.g., https://api.openai.com/v1)",
            },
            llm_api_key: {
              type: "string",
              description: "API key for the agent LLM",
            },
            llm_model: {
              type: "string",
              description: "Model name for the agent LLM (e.g., gpt-4o)",
            },
          },
          required: ["model_id", "worker_id"],
        },
      },
      {
        name: "get_tuning_job_status",
        description:
          "Get the status of an Auto-Tuning job (one-time check only). WARNING: Do NOT use this to monitor progress - use wait_for_tuning_job instead which will block until completion.",
        inputSchema: {
          type: "object",
          properties: {
            job_id: {
              type: "number",
              description: "ID of the tuning job",
            },
          },
          required: ["job_id"],
        },
      },
      {
        name: "wait_for_tuning_job",
        description:
          "Wait for an Auto-Tuning job to complete. This blocks until the job finishes (completed/failed/cancelled) or times out. Use this instead of repeatedly calling get_tuning_job_status. Model loading takes time, so set an appropriate timeout.",
        inputSchema: {
          type: "object",
          properties: {
            job_id: {
              type: "number",
              description: "ID of the tuning job to wait for",
            },
            timeout_seconds: {
              type: "number",
              description: "Maximum time to wait in seconds (default: 600 = 10 minutes)",
            },
          },
          required: ["job_id"],
        },
      },
      {
        name: "list_tuning_jobs",
        description:
          "List all Auto-Tuning jobs with their status.",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "query_knowledge_base",
        description:
          "Query the performance knowledge base for historical benchmark results and recommended configurations.",
        inputSchema: {
          type: "object",
          properties: {
            model_name: {
              type: "string",
              description: "Filter by model name pattern",
            },
            gpu_model: {
              type: "string",
              description: "Filter by GPU model pattern",
            },
            limit: {
              type: "number",
              description: "Maximum results to return (default: 10)",
            },
          },
          required: [],
        },
      },
    ],
  };
});

/**
 * Execute tools
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      // ============== Query Tools ==============
      case "get_system_status": {
        const [workers, containers, deployments, models] = await Promise.all([
          client.getWorkers(),
          client.getContainers(),
          client.getDeployments(),
          client.getModels(),
        ]);
        return {
          content: [
            {
              type: "text",
              text: formatSystemStatus(workers, containers, deployments, models),
            },
          ],
        };
      }

      case "list_workers": {
        const workers = await client.getWorkers();
        return {
          content: [
            {
              type: "text",
              text: formatWorkers(workers),
            },
          ],
        };
      }

      case "list_containers": {
        const containers = await client.getContainers();
        let filtered = containers;

        if (args?.worker_id) {
          filtered = filtered.filter((c: any) =>
            c.worker?.id === args.worker_id || c.worker_id === args.worker_id
          );
        }

        if (args?.status && args.status !== "all") {
          filtered = filtered.filter((c: any) => {
            const s = c.status?.toLowerCase() || "";
            if (args.status === "running") {
              return s.includes("running") || s.includes("up");
            }
            return s.includes(String(args.status));
          });
        }

        return {
          content: [
            {
              type: "text",
              text: formatContainers(filtered),
            },
          ],
        };
      }

      case "list_deployments": {
        const deployments = await client.getDeployments();
        let filtered = deployments;

        if (args?.status && args.status !== "all") {
          filtered = filtered.filter((d: any) =>
            d.status?.toLowerCase() === String(args.status).toLowerCase()
          );
        }

        return {
          content: [
            {
              type: "text",
              text: formatDeployments(filtered),
            },
          ],
        };
      }

      case "list_models": {
        const models = await client.getModels();
        let filtered = models;

        if (args?.source) {
          filtered = filtered.filter((m: any) =>
            m.source?.toLowerCase() === String(args.source).toLowerCase()
          );
        }

        return {
          content: [
            {
              type: "text",
              text: formatModels(filtered),
            },
          ],
        };
      }

      case "get_gpu_status": {
        const workers = await client.getWorkers();
        let filtered = workers;

        if (args?.worker_id) {
          filtered = filtered.filter((w: any) => w.id === args.worker_id);
        }

        const lines: string[] = ["# GPU Status\n"];
        for (const worker of filtered) {
          lines.push(`## ${worker.name} (${worker.status})`);
          if (worker.gpu_info && worker.gpu_info.length > 0) {
            for (const gpu of worker.gpu_info) {
              const usedGB = (gpu.memory_used / 1024).toFixed(1);
              const totalGB = (gpu.memory_total / 1024).toFixed(1);
              const freeGB = ((gpu.memory_total - gpu.memory_used) / 1024).toFixed(1);
              lines.push(`- GPU ${gpu.index}: ${gpu.name}`);
              lines.push(`  - Memory: ${usedGB}GB used / ${freeGB}GB free / ${totalGB}GB total`);
              lines.push(`  - Utilization: ${gpu.utilization_gpu}%`);
            }
          } else {
            lines.push("- No GPU information available");
          }
          lines.push("");
        }

        return {
          content: [
            {
              type: "text",
              text: lines.join("\n"),
            },
          ],
        };
      }

      // ============== Model Management Tools ==============
      case "add_model": {
        if (!args?.name || !args?.source) {
          throw new Error("name and source are required");
        }

        const model = await client.addModel(
          String(args.name),
          String(args.source),
          args.parameters ? String(args.parameters) : undefined,
          args.quantization ? String(args.quantization) : undefined
        );

        return {
          content: [
            {
              type: "text",
              text: `Successfully added model!\n\nModel ID: ${model.id}\nName: ${model.name}\nSource: ${model.source}`,
            },
          ],
        };
      }

      case "delete_model": {
        if (!args?.model_id) {
          throw new Error("model_id is required");
        }

        await client.deleteModel(Number(args.model_id));

        return {
          content: [
            {
              type: "text",
              text: `Successfully deleted model ${args.model_id}`,
            },
          ],
        };
      }

      // ============== Deployment Tools ==============
      case "deploy_model": {
        if (!args?.model_id || !args?.worker_id) {
          throw new Error("model_id and worker_id are required");
        }

        const result = await client.deployModel(
          Number(args.model_id),
          Number(args.worker_id),
          args.name as string | undefined,
          args.gpu_indexes as number[] | undefined,
          args.backend as string | undefined
        );

        return {
          content: [
            {
              type: "text",
              text: `Successfully started deployment!\n\nDeployment ID: ${result.id}\nName: ${result.name}\nStatus: ${result.status}\n\nThe model is being deployed. Use list_deployments to check status.`,
            },
          ],
        };
      }

      case "stop_deployment": {
        if (!args?.deployment_id) {
          throw new Error("deployment_id is required");
        }

        await client.stopDeployment(Number(args.deployment_id));

        return {
          content: [
            {
              type: "text",
              text: `Successfully stopped deployment ${args.deployment_id}`,
            },
          ],
        };
      }

      case "start_deployment": {
        if (!args?.deployment_id) {
          throw new Error("deployment_id is required");
        }

        await client.startDeployment(Number(args.deployment_id));

        return {
          content: [
            {
              type: "text",
              text: `Successfully started deployment ${args.deployment_id}`,
            },
          ],
        };
      }

      case "delete_deployment": {
        if (!args?.deployment_id) {
          throw new Error("deployment_id is required");
        }

        await client.deleteDeployment(Number(args.deployment_id));

        return {
          content: [
            {
              type: "text",
              text: `Successfully deleted deployment ${args.deployment_id}`,
            },
          ],
        };
      }

      // ============== Container Tools ==============
      case "stop_container": {
        if (!args?.container_name || !args?.worker_id) {
          throw new Error("container_name and worker_id are required");
        }

        await client.stopContainer(String(args.container_name), Number(args.worker_id));

        return {
          content: [
            {
              type: "text",
              text: `Successfully stopped container "${args.container_name}"`,
            },
          ],
        };
      }

      case "remove_container": {
        if (!args?.container_name || !args?.worker_id) {
          throw new Error("container_name and worker_id are required");
        }

        await client.removeContainer(
          String(args.container_name),
          Number(args.worker_id),
          args.force as boolean | undefined
        );

        return {
          content: [
            {
              type: "text",
              text: `Successfully removed container "${args.container_name}"`,
            },
          ],
        };
      }

      // ============== API Key Tools ==============
      case "list_api_keys": {
        const apiKeysData = await client.getApiKeys();
        return {
          content: [
            {
              type: "text",
              text: formatApiKeys(apiKeysData),
            },
          ],
        };
      }

      case "create_api_key": {
        if (!args?.name) {
          throw new Error("name is required");
        }

        const apiKey = await client.createApiKey(
          String(args.name),
          args.description ? String(args.description) : undefined,
          args.expires_in_days ? Number(args.expires_in_days) : undefined
        );

        return {
          content: [
            {
              type: "text",
              text: `Successfully created API key!\n\nID: ${apiKey.id}\nName: ${apiKey.name}\nAccess Key: ${apiKey.access_key}\nFull Key: ${apiKey.api_key}\n\n**IMPORTANT:** Save the full API key now! It will not be shown again.`,
            },
          ],
        };
      }

      case "delete_api_key": {
        if (!args?.api_key_id) {
          throw new Error("api_key_id is required");
        }

        await client.deleteApiKey(Number(args.api_key_id));

        return {
          content: [
            {
              type: "text",
              text: `Successfully deleted API key ${args.api_key_id}`,
            },
          ],
        };
      }

      // ============== Docker Image Tools ==============
      case "list_images": {
        const images = await client.getImages(
          args?.worker_id ? Number(args.worker_id) : undefined,
          args?.repository ? String(args.repository) : undefined
        );
        return {
          content: [
            {
              type: "text",
              text: formatImages(images),
            },
          ],
        };
      }

      case "pull_image": {
        if (!args?.worker_id || !args?.image) {
          throw new Error("worker_id and image are required");
        }

        const result = await client.pullImage(Number(args.worker_id), String(args.image));

        return {
          content: [
            {
              type: "text",
              text: `Successfully pulled image "${args.image}"\n\n${JSON.stringify(result, null, 2)}`,
            },
          ],
        };
      }

      case "delete_image": {
        if (!args?.image_id || !args?.worker_id) {
          throw new Error("image_id and worker_id are required");
        }

        await client.deleteImage(
          String(args.image_id),
          Number(args.worker_id),
          args.force as boolean | undefined
        );

        return {
          content: [
            {
              type: "text",
              text: `Successfully deleted image "${args.image_id}"`,
            },
          ],
        };
      }

      // ============== Storage Tools ==============
      case "list_storage_volumes": {
        const volumes = await client.getStorageVolumes(
          args?.worker_id ? Number(args.worker_id) : undefined
        );
        return {
          content: [
            {
              type: "text",
              text: formatStorageVolumes(volumes),
            },
          ],
        };
      }

      case "get_disk_usage": {
        const diskUsage = await client.getDiskUsage(
          args?.worker_id ? Number(args.worker_id) : undefined
        );
        return {
          content: [
            {
              type: "text",
              text: formatDiskUsage(diskUsage),
            },
          ],
        };
      }

      case "delete_storage_volume": {
        if (!args?.volume_name || !args?.worker_id) {
          throw new Error("volume_name and worker_id are required");
        }

        await client.deleteStorageVolume(
          String(args.volume_name),
          Number(args.worker_id),
          args.force as boolean | undefined
        );

        return {
          content: [
            {
              type: "text",
              text: `Successfully deleted volume "${args.volume_name}"`,
            },
          ],
        };
      }

      case "prune_storage": {
        const results = await client.pruneStorage(
          args?.worker_id ? Number(args.worker_id) : undefined,
          args?.images !== false,
          args?.containers !== false,
          args?.volumes === true,
          args?.build_cache !== false
        );

        const formatSize = (bytes: number) => {
          if (bytes >= 1024 * 1024 * 1024) {
            return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
          }
          return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
        };

        const lines = ["# Storage Pruned Successfully\n"];
        for (const r of results) {
          lines.push(`## ${r.worker_name || "Worker"}`);
          lines.push(`- Images deleted: ${r.images_deleted || 0}`);
          lines.push(`- Containers deleted: ${r.containers_deleted || 0}`);
          lines.push(`- Volumes deleted: ${r.volumes_deleted || 0}`);
          lines.push(`- Build cache deleted: ${r.build_cache_deleted || 0}`);
          lines.push(`- Space reclaimed: ${formatSize(r.space_reclaimed || 0)}`);
          lines.push("");
        }

        return {
          content: [
            {
              type: "text",
              text: lines.join("\n"),
            },
          ],
        };
      }

      // ============== Web Search Tools ==============
      case "web_search": {
        if (!args?.query) {
          throw new Error("query is required");
        }

        const searchResults = await webSearch(
          String(args.query),
          args.max_results ? Number(args.max_results) : 5
        );

        return {
          content: [
            {
              type: "text",
              text: searchResults,
            },
          ],
        };
      }

      case "search_llm_config": {
        if (!args?.gpu_model || !args?.llm_model) {
          throw new Error("gpu_model and llm_model are required");
        }

        const configResults = await searchLLMConfig(
          String(args.gpu_model),
          String(args.llm_model),
          args.backend ? String(args.backend) : "vLLM"
        );

        return {
          content: [
            {
              type: "text",
              text: configResults,
            },
          ],
        };
      }

      // ============== Auto-Tuning & Benchmark Tools ==============
      case "run_benchmark": {
        if (!args?.deployment_id) {
          throw new Error("deployment_id is required");
        }

        const benchmarkResults = await runBenchmark(client, {
          deploymentId: Number(args.deployment_id),
          durationSeconds: args.duration_seconds ? Number(args.duration_seconds) : undefined,
          inputLength: args.input_length ? Number(args.input_length) : undefined,
          outputLength: args.output_length ? Number(args.output_length) : undefined,
          concurrency: args.concurrency ? Number(args.concurrency) : undefined,
        });

        return {
          content: [
            {
              type: "text",
              text: benchmarkResults,
            },
          ],
        };
      }

      case "start_auto_tuning": {
        if (!args?.model_id || !args?.worker_id) {
          throw new Error("model_id and worker_id are required");
        }

        const tuningResults = await startAutoTuning(
          client,
          Number(args.model_id),
          Number(args.worker_id),
          {
            engines: args.engines as string[] | undefined,
            tensorParallelSizes: args.tensor_parallel_sizes as number[] | undefined,
            gpuMemoryUtilizations: args.gpu_memory_utilizations as number[] | undefined,
            maxModelLengths: args.max_model_lengths as number[] | undefined,
            concurrencyLevels: args.concurrency_levels as number[] | undefined,
            llmBaseUrl: args.llm_base_url as string | undefined,
            llmApiKey: args.llm_api_key as string | undefined,
            llmModel: args.llm_model as string | undefined,
          }
        );

        return {
          content: [
            {
              type: "text",
              text: tuningResults,
            },
          ],
        };
      }

      case "get_tuning_job_status": {
        if (!args?.job_id) {
          throw new Error("job_id is required");
        }

        const statusResults = await getTuningJobStatus(client, Number(args.job_id));

        return {
          content: [
            {
              type: "text",
              text: statusResults,
            },
          ],
        };
      }

      case "wait_for_tuning_job": {
        if (!args?.job_id) {
          throw new Error("job_id is required");
        }

        const waitResults = await waitForTuningJob(
          client,
          Number(args.job_id),
          args.timeout_seconds ? Number(args.timeout_seconds) : 600
        );

        return {
          content: [
            {
              type: "text",
              text: waitResults,
            },
          ],
        };
      }

      case "list_tuning_jobs": {
        const jobs = await client.listTuningJobs();

        if (jobs.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: "## Auto-Tuning Jobs\n\nNo tuning jobs found.",
              },
            ],
          };
        }

        let output = "## Auto-Tuning Jobs\n\n";
        output += "| ID | Model | Status | Progress | Created |\n";
        output += "|----|-------|--------|----------|--------|\n";

        for (const job of jobs) {
          const progress = job.progress
            ? `${job.progress.configs_tested || 0}/${job.progress.configs_total || "?"}`
            : "-";
          output += `| ${job.id} | ${job.model_name || "Unknown"} | ${job.status} | ${progress} | ${new Date(job.created_at).toLocaleString()} |\n`;
        }

        return {
          content: [
            {
              type: "text",
              text: output,
            },
          ],
        };
      }

      case "query_knowledge_base": {
        const results = await client.queryKnowledgeBase({
          model_name: args?.model_name ? String(args.model_name) : undefined,
          gpu_model: args?.gpu_model ? String(args.gpu_model) : undefined,
          limit: args?.limit ? Number(args.limit) : 10,
        });

        if (results.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: "## Knowledge Base\n\nNo matching records found.",
              },
            ],
          };
        }

        let output = "## Knowledge Base Results\n\n";
        output += "| Model | GPU | Engine | TPS | TTFT | TPOT |\n";
        output += "|-------|-----|--------|-----|------|------|\n";

        for (const r of results) {
          output += `| ${r.model_name} | ${r.gpu_count}x ${r.gpu_model} | ${r.engine} | ${r.throughput_tps.toFixed(1)} | ${r.ttft_ms.toFixed(0)}ms | ${r.tpot_ms.toFixed(1)}ms |\n`;
        }

        return {
          content: [
            {
              type: "text",
              text: output,
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      content: [
        {
          type: "text",
          text: `Error: ${message}`,
        },
      ],
      isError: true,
    };
  }
});

/**
 * Start the server
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("LMStack MCP Server running on stdio");
  console.error(`API URL: ${LMSTACK_API_URL}`);
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
