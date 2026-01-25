/**
 * Chat Tools Definition
 *
 * Tools that the AI assistant can call to interact with LMStack.
 * Includes confirmation flow for dangerous operations.
 */
import { api } from "../../api/client";
import type { ChatModelConfig } from "./types";

/**
 * Tool definition for OpenAI-compatible API
 */
export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, any>;
      required?: string[];
    };
  };
}

/**
 * Tool call from LLM response
 */
export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

/**
 * Tool execution result
 */
export interface ToolResult {
  tool_call_id: string;
  role: "tool";
  content: string;
}

/**
 * Tool metadata for UI display
 */
export interface ToolMeta {
  name: string;
  displayName: string;
  description: string;
  category: "query" | "action";
  dangerous: boolean;
  icon: string;
}

/**
 * Pending tool execution for confirmation
 */
export interface PendingToolExecution {
  toolCall: ToolCall;
  parsedArgs: Record<string, any>;
  meta: ToolMeta;
}

/**
 * Tool metadata registry
 */
export const TOOL_META: Record<string, ToolMeta> = {
  // Query tools (no confirmation needed)
  get_system_status: {
    name: "get_system_status",
    displayName: "Get System Status",
    description: "Query complete system status",
    category: "query",
    dangerous: false,
    icon: "dashboard",
  },
  list_workers: {
    name: "list_workers",
    displayName: "List Workers",
    description: "Query all worker nodes",
    category: "query",
    dangerous: false,
    icon: "cluster",
  },
  list_containers: {
    name: "list_containers",
    displayName: "List Containers",
    description: "Query Docker containers",
    category: "query",
    dangerous: false,
    icon: "container",
  },
  list_deployments: {
    name: "list_deployments",
    displayName: "List Deployments",
    description: "Query model deployments",
    category: "query",
    dangerous: false,
    icon: "rocket",
  },
  list_models: {
    name: "list_models",
    displayName: "List Models",
    description: "Query available models",
    category: "query",
    dangerous: false,
    icon: "robot",
  },
  get_gpu_status: {
    name: "get_gpu_status",
    displayName: "Get GPU Status",
    description: "Query GPU usage",
    category: "query",
    dangerous: false,
    icon: "thunderbolt",
  },

  // Action tools (confirmation needed)
  add_model: {
    name: "add_model",
    displayName: "Add Model",
    description: "Add a new model to the system",
    category: "action",
    dangerous: false,
    icon: "plus",
  },
  delete_model: {
    name: "delete_model",
    displayName: "Delete Model",
    description: "Delete a model from the system",
    category: "action",
    dangerous: true,
    icon: "delete",
  },
  deploy_model: {
    name: "deploy_model",
    displayName: "Deploy Model",
    description: "Deploy a model to a worker",
    category: "action",
    dangerous: false,
    icon: "rocket",
  },
  stop_deployment: {
    name: "stop_deployment",
    displayName: "Stop Deployment",
    description: "Stop a running deployment",
    category: "action",
    dangerous: true,
    icon: "pause-circle",
  },
  start_deployment: {
    name: "start_deployment",
    displayName: "Start Deployment",
    description: "Start a stopped deployment",
    category: "action",
    dangerous: false,
    icon: "play-circle",
  },
  delete_deployment: {
    name: "delete_deployment",
    displayName: "Delete Deployment",
    description: "Permanently delete a deployment",
    category: "action",
    dangerous: true,
    icon: "delete",
  },
  stop_container: {
    name: "stop_container",
    displayName: "Stop Container",
    description: "Stop a Docker container",
    category: "action",
    dangerous: true,
    icon: "stop",
  },
  remove_container: {
    name: "remove_container",
    displayName: "Remove Container",
    description: "Remove a Docker container",
    category: "action",
    dangerous: true,
    icon: "delete",
  },

  // API Key tools
  list_api_keys: {
    name: "list_api_keys",
    displayName: "List API Keys",
    description: "Query all API keys",
    category: "query",
    dangerous: false,
    icon: "key",
  },
  create_api_key: {
    name: "create_api_key",
    displayName: "Create API Key",
    description: "Create a new API key",
    category: "action",
    dangerous: false,
    icon: "plus",
  },
  delete_api_key: {
    name: "delete_api_key",
    displayName: "Delete API Key",
    description: "Delete an API key",
    category: "action",
    dangerous: true,
    icon: "delete",
  },

  // Docker Image tools
  list_images: {
    name: "list_images",
    displayName: "List Images",
    description: "Query Docker images",
    category: "query",
    dangerous: false,
    icon: "container",
  },
  pull_image: {
    name: "pull_image",
    displayName: "Pull Image",
    description: "Pull a Docker image from registry",
    category: "action",
    dangerous: false,
    icon: "download",
  },
  delete_image: {
    name: "delete_image",
    displayName: "Delete Image",
    description: "Delete a Docker image",
    category: "action",
    dangerous: true,
    icon: "delete",
  },

  // Storage tools
  list_storage_volumes: {
    name: "list_storage_volumes",
    displayName: "List Storage Volumes",
    description: "Query Docker storage volumes",
    category: "query",
    dangerous: false,
    icon: "database",
  },
  get_disk_usage: {
    name: "get_disk_usage",
    displayName: "Get Disk Usage",
    description: "Query disk usage statistics",
    category: "query",
    dangerous: false,
    icon: "pie-chart",
  },
  delete_storage_volume: {
    name: "delete_storage_volume",
    displayName: "Delete Storage Volume",
    description: "Delete a Docker storage volume",
    category: "action",
    dangerous: true,
    icon: "delete",
  },
  prune_storage: {
    name: "prune_storage",
    displayName: "Prune Storage",
    description: "Clean up unused Docker resources",
    category: "action",
    dangerous: true,
    icon: "clear",
  },

  // Auto-Tuning tools
  list_tuning_jobs: {
    name: "list_tuning_jobs",
    displayName: "List Tuning Jobs",
    description: "Query all auto-tuning jobs",
    category: "query",
    dangerous: false,
    icon: "thunderbolt",
  },
  start_auto_tuning: {
    name: "start_auto_tuning",
    displayName: "Start Auto-Tuning",
    description: "Start a new auto-tuning job",
    category: "action",
    dangerous: false,
    icon: "experiment",
  },
  get_tuning_job: {
    name: "get_tuning_job",
    displayName: "Get Tuning Job",
    description: "Get details of a tuning job",
    category: "query",
    dangerous: false,
    icon: "info-circle",
  },
  cancel_tuning_job: {
    name: "cancel_tuning_job",
    displayName: "Cancel Tuning Job",
    description: "Cancel a running tuning job",
    category: "action",
    dangerous: true,
    icon: "stop",
  },
  query_knowledge_base: {
    name: "query_knowledge_base",
    displayName: "Query Knowledge Base",
    description: "Query performance knowledge base",
    category: "query",
    dangerous: false,
    icon: "database",
  },
  run_benchmark: {
    name: "run_benchmark",
    displayName: "Run Benchmark",
    description: "Run performance benchmark on a deployment",
    category: "action",
    dangerous: false,
    icon: "bar-chart",
  },
};

/**
 * Check if a tool requires confirmation
 */
export function requiresConfirmation(toolName: string): boolean {
  const meta = TOOL_META[toolName];
  return meta?.category === "action";
}

/**
 * Get tool metadata
 */
export function getToolMeta(toolName: string): ToolMeta {
  return (
    TOOL_META[toolName] || {
      name: toolName,
      displayName: toolName,
      description: "Unknown tool",
      category: "action",
      dangerous: false,
      icon: "question",
    }
  );
}

/**
 * Available tools for the AI assistant
 */
export const CHAT_TOOLS: ToolDefinition[] = [
  // ============== Query Tools ==============
  {
    type: "function",
    function: {
      name: "get_system_status",
      description:
        "Get complete LMStack system status including workers, GPUs, containers, and deployments. Call this to get the latest system information.",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_workers",
      description:
        "List all worker nodes with their GPU status, memory usage, and availability.",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_containers",
      description: "List all Docker containers running across all workers.",
      parameters: {
        type: "object",
        properties: {
          status: {
            type: "string",
            description: "Filter by status: running, stopped, or all",
            enum: ["running", "stopped", "all"],
          },
          worker_id: {
            type: "number",
            description: "Optional: Filter by specific worker ID",
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_deployments",
      description: "List all model deployments with their status.",
      parameters: {
        type: "object",
        properties: {
          status: {
            type: "string",
            description: "Filter by status: running, stopped, or all",
            enum: ["running", "stopped", "all"],
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_models",
      description: "List all available models that can be deployed.",
      parameters: {
        type: "object",
        properties: {
          source: {
            type: "string",
            description: "Filter by source",
            enum: ["huggingface", "ollama", "local"],
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_gpu_status",
      description:
        "Get detailed GPU status including memory usage, utilization, and temperature for all workers.",
      parameters: {
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
  },

  // ============== Model Management Tools ==============
  {
    type: "function",
    function: {
      name: "add_model",
      description:
        "Add a new model to the system. Supports HuggingFace and Ollama models.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description:
              "Model name/identifier (e.g., 'Qwen/Qwen2.5-7B-Instruct' for HuggingFace, 'llama3.2' for Ollama)",
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
            description:
              "Optional: Quantization format (e.g., 'GPTQ', 'AWQ', 'GGUF')",
          },
        },
        required: ["name", "source"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "delete_model",
      description:
        "Delete a model from the system. This will NOT delete any deployments using this model.",
      parameters: {
        type: "object",
        properties: {
          model_id: {
            type: "number",
            description:
              "ID of the model to delete (use list_models to find IDs)",
          },
        },
        required: ["model_id"],
      },
    },
  },

  // ============== Deployment Tools ==============
  {
    type: "function",
    function: {
      name: "deploy_model",
      description:
        "Deploy a model to a worker. This will start the model inference service.",
      parameters: {
        type: "object",
        properties: {
          model_id: {
            type: "number",
            description:
              "ID of the model to deploy (use list_models to find IDs)",
          },
          worker_id: {
            type: "number",
            description:
              "ID of the worker to deploy to (use list_workers to find IDs)",
          },
          gpu_ids: {
            type: "array",
            items: { type: "number" },
            description:
              "Optional: Specific GPU indices to use. If not provided, GPUs will be auto-selected.",
          },
        },
        required: ["model_id", "worker_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "stop_deployment",
      description: "Stop a running model deployment.",
      parameters: {
        type: "object",
        properties: {
          deployment_id: {
            type: "number",
            description:
              "ID of the deployment to stop (use list_deployments to find IDs)",
          },
        },
        required: ["deployment_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "start_deployment",
      description: "Start a stopped model deployment.",
      parameters: {
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
  },
  {
    type: "function",
    function: {
      name: "delete_deployment",
      description:
        "Delete a model deployment completely. This cannot be undone.",
      parameters: {
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
  },

  // ============== Container Tools ==============
  {
    type: "function",
    function: {
      name: "stop_container",
      description:
        "Stop a running Docker container. If you don't know the worker_id, call list_containers first to find it.",
      parameters: {
        type: "object",
        properties: {
          container_name: {
            type: "string",
            description:
              "Name of the container to stop (e.g., 'lmstack-llama')",
          },
          worker_id: {
            type: "number",
            description:
              "ID of the worker where the container is running. Use list_containers to find this.",
          },
        },
        required: ["container_name", "worker_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "remove_container",
      description:
        "Remove/delete a Docker container. If you don't know the worker_id, call list_containers first to find it.",
      parameters: {
        type: "object",
        properties: {
          container_name: {
            type: "string",
            description: "Name of the container to remove",
          },
          worker_id: {
            type: "number",
            description:
              "ID of the worker where the container is located. Use list_containers to find this.",
          },
          force: {
            type: "boolean",
            description: "Force remove even if running (default: false)",
          },
        },
        required: ["container_name", "worker_id"],
      },
    },
  },

  // ============== API Key Tools ==============
  {
    type: "function",
    function: {
      name: "list_api_keys",
      description:
        "List all API keys in the system with their usage statistics.",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "create_api_key",
      description: "Create a new API key for accessing the LMStack API.",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description:
              "Name for the API key (e.g., 'production-key', 'test-key')",
          },
          description: {
            type: "string",
            description: "Optional description for the API key",
          },
          expires_in_days: {
            type: "number",
            description:
              "Optional: Number of days until the key expires. If not set, the key never expires.",
          },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "delete_api_key",
      description: "Delete an API key from the system.",
      parameters: {
        type: "object",
        properties: {
          api_key_id: {
            type: "number",
            description:
              "ID of the API key to delete (use list_api_keys to find IDs)",
          },
        },
        required: ["api_key_id"],
      },
    },
  },

  // ============== Docker Image Tools ==============
  {
    type: "function",
    function: {
      name: "list_images",
      description: "List all Docker images across all workers.",
      parameters: {
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
  },
  {
    type: "function",
    function: {
      name: "pull_image",
      description: "Pull a Docker image from a registry to a worker.",
      parameters: {
        type: "object",
        properties: {
          worker_id: {
            type: "number",
            description: "ID of the worker to pull the image to",
          },
          image: {
            type: "string",
            description:
              "Image reference (e.g., 'nginx:latest', 'python:3.11')",
          },
        },
        required: ["worker_id", "image"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "delete_image",
      description: "Delete a Docker image from a worker.",
      parameters: {
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
            description:
              "Force removal even if image is in use (default: false)",
          },
        },
        required: ["image_id", "worker_id"],
      },
    },
  },

  // ============== Storage Tools ==============
  {
    type: "function",
    function: {
      name: "list_storage_volumes",
      description: "List all Docker storage volumes across all workers.",
      parameters: {
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
  },
  {
    type: "function",
    function: {
      name: "get_disk_usage",
      description:
        "Get Docker disk usage statistics including images, containers, volumes, and build cache.",
      parameters: {
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
  },
  {
    type: "function",
    function: {
      name: "delete_storage_volume",
      description: "Delete a Docker storage volume from a worker.",
      parameters: {
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
  },
  {
    type: "function",
    function: {
      name: "prune_storage",
      description:
        "Clean up unused Docker resources (images, containers, volumes, build cache) to free disk space.",
      parameters: {
        type: "object",
        properties: {
          worker_id: {
            type: "number",
            description:
              "Optional: Only prune on specific worker. If not set, prunes on all workers.",
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
  },

  // ============== Auto-Tuning Tools ==============
  {
    type: "function",
    function: {
      name: "list_tuning_jobs",
      description: "List all auto-tuning jobs with their status and progress.",
      parameters: {
        type: "object",
        properties: {
          status: {
            type: "string",
            description: "Filter by status",
            enum: [
              "pending",
              "analyzing",
              "querying_kb",
              "exploring",
              "benchmarking",
              "completed",
              "failed",
              "cancelled",
            ],
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "start_auto_tuning",
      description:
        "Start a new auto-tuning job to find the best deployment configuration for a model. The agent will analyze the environment, query the knowledge base, explore configuration space, run benchmarks, and find the optimal settings.",
      parameters: {
        type: "object",
        properties: {
          model_id: {
            type: "number",
            description:
              "ID of the model to tune (use list_models to find IDs)",
          },
          worker_id: {
            type: "number",
            description:
              "ID of the worker to use for tuning (use list_workers to find IDs)",
          },
          optimization_target: {
            type: "string",
            description: "What to optimize for",
            enum: ["throughput", "latency", "cost", "balanced"],
          },
        },
        required: ["model_id", "worker_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_tuning_job",
      description:
        "Get detailed information about a specific tuning job including progress, best configuration, and all results.",
      parameters: {
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
  },
  {
    type: "function",
    function: {
      name: "cancel_tuning_job",
      description: "Cancel a running auto-tuning job.",
      parameters: {
        type: "object",
        properties: {
          job_id: {
            type: "number",
            description: "ID of the tuning job to cancel",
          },
        },
        required: ["job_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "query_knowledge_base",
      description:
        "Query the performance knowledge base to find similar configurations and their benchmark results. This uses transfer learning from previous tuning results.",
      parameters: {
        type: "object",
        properties: {
          model_name: {
            type: "string",
            description: "Model name pattern to match (e.g., 'Qwen', 'Llama')",
          },
          model_family: {
            type: "string",
            description: "Model family: Qwen, Llama, Mistral, etc.",
          },
          gpu_model: {
            type: "string",
            description: "GPU model pattern (e.g., 'RTX 4090', 'A100')",
          },
          optimization_target: {
            type: "string",
            description: "Optimization target for scoring",
            enum: ["throughput", "latency", "cost", "balanced"],
          },
          limit: {
            type: "number",
            description: "Maximum number of results to return (default: 10)",
          },
        },
        required: [],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_benchmark",
      description:
        "Run a performance benchmark on a deployment to measure throughput, latency, and resource usage.",
      parameters: {
        type: "object",
        properties: {
          deployment_id: {
            type: "number",
            description:
              "ID of the deployment to benchmark (use list_deployments to find IDs)",
          },
          test_type: {
            type: "string",
            description: "Type of benchmark test",
            enum: ["throughput", "latency"],
          },
          duration_seconds: {
            type: "number",
            description: "Test duration in seconds (10-600, default: 60)",
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
            description: "Number of concurrent requests (1-64, default: 1)",
          },
        },
        required: ["deployment_id"],
      },
    },
  },
];

/**
 * Execute a tool call and return the result
 * @param toolCall - The tool call to execute
 * @param modelConfig - Optional model config for tools that need LLM access (like auto-tuning)
 */
export async function executeTool(
  toolCall: ToolCall,
  modelConfig?: ChatModelConfig,
): Promise<ToolResult> {
  const { name, arguments: argsStr } = toolCall.function;
  let args: Record<string, any> = {};

  try {
    args = JSON.parse(argsStr);
  } catch {
    return {
      tool_call_id: toolCall.id,
      role: "tool",
      content: `Error: Invalid arguments JSON: ${argsStr}`,
    };
  }

  try {
    let result: string;

    switch (name) {
      // Query tools
      case "get_system_status":
        result = await getSystemStatus();
        break;

      case "list_workers":
        result = await listWorkers();
        break;

      case "list_containers":
        result = await listContainers(args.status, args.worker_id);
        break;

      case "list_deployments":
        result = await listDeployments(args.status);
        break;

      case "list_models":
        result = await listModels(args.source);
        break;

      case "get_gpu_status":
        result = await getGpuStatus(args.worker_id);
        break;

      // Model management tools
      case "add_model":
        result = await addModel(
          args.name,
          args.source,
          args.parameters,
          args.quantization,
        );
        break;

      case "delete_model":
        result = await deleteModel(args.model_id);
        break;

      // Deployment tools
      case "deploy_model":
        result = await deployModel(args.model_id, args.worker_id, args.gpu_ids);
        break;

      case "stop_deployment":
        result = await stopDeployment(args.deployment_id);
        break;

      case "start_deployment":
        result = await startDeployment(args.deployment_id);
        break;

      case "delete_deployment":
        result = await deleteDeployment(args.deployment_id);
        break;

      // Container tools
      case "stop_container":
        result = await stopContainer(args.container_name, args.worker_id);
        break;

      case "remove_container":
        result = await removeContainer(
          args.container_name,
          args.worker_id,
          args.force,
        );
        break;

      // API Key tools
      case "list_api_keys":
        result = await listApiKeys();
        break;

      case "create_api_key":
        result = await createApiKey(
          args.name,
          args.description,
          args.expires_in_days,
        );
        break;

      case "delete_api_key":
        result = await deleteApiKey(args.api_key_id);
        break;

      // Docker Image tools
      case "list_images":
        result = await listImages(args.worker_id, args.repository);
        break;

      case "pull_image":
        result = await pullImage(args.worker_id, args.image);
        break;

      case "delete_image":
        result = await deleteImage(args.image_id, args.worker_id, args.force);
        break;

      // Storage tools
      case "list_storage_volumes":
        result = await listStorageVolumes(args.worker_id);
        break;

      case "get_disk_usage":
        result = await getDiskUsage(args.worker_id);
        break;

      case "delete_storage_volume":
        result = await deleteStorageVolume(
          args.volume_name,
          args.worker_id,
          args.force,
        );
        break;

      case "prune_storage":
        result = await pruneStorage(
          args.worker_id,
          args.images,
          args.containers,
          args.volumes,
          args.build_cache,
        );
        break;

      // Auto-Tuning tools
      case "list_tuning_jobs":
        result = await listTuningJobs(args.status);
        break;

      case "start_auto_tuning":
        result = await startAutoTuning(
          args.model_id,
          args.worker_id,
          args.optimization_target,
          modelConfig,
        );
        break;

      case "get_tuning_job":
        result = await getTuningJob(args.job_id);
        break;

      case "cancel_tuning_job":
        result = await cancelTuningJob(args.job_id);
        break;

      case "query_knowledge_base":
        result = await queryKnowledgeBase(
          args.model_name,
          args.model_family,
          args.gpu_model,
          args.optimization_target,
          args.limit,
        );
        break;

      case "run_benchmark":
        result = await runBenchmark(
          args.deployment_id,
          args.test_type,
          args.duration_seconds,
          args.input_length,
          args.output_length,
          args.concurrency,
        );
        break;

      default:
        result = `Unknown tool: ${name}`;
    }

    return {
      tool_call_id: toolCall.id,
      role: "tool",
      content: result,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      tool_call_id: toolCall.id,
      role: "tool",
      content: `Error executing ${name}: ${message}`,
    };
  }
}

// ============================================================================
// Tool Implementations
// ============================================================================

async function getSystemStatus(): Promise<string> {
  const [workers, containers, deployments, models] = await Promise.all([
    api.get("/workers").then((r) => r.data.items || []),
    api.get("/containers").then((r) => r.data.items || []),
    api.get("/deployments").then((r) => r.data.items || []),
    api.get("/models").then((r) => r.data.items || []),
  ]);

  const onlineWorkers = workers.filter((w: any) => w.status === "online");
  const runningContainers = containers.filter(
    (c: any) =>
      c.status?.toLowerCase().includes("running") ||
      c.status?.toLowerCase().includes("up"),
  );
  const runningDeployments = deployments.filter(
    (d: any) => d.status === "running",
  );

  let totalGpuMem = 0,
    usedGpuMem = 0;
  for (const w of workers) {
    for (const g of w.gpu_info || []) {
      totalGpuMem += g.memory_total || 0;
      usedGpuMem += g.memory_used || 0;
    }
  }

  return JSON.stringify(
    {
      summary: {
        workers: `${onlineWorkers.length}/${workers.length} online`,
        containers: `${runningContainers.length}/${containers.length} running`,
        deployments: `${runningDeployments.length}/${deployments.length} running`,
        models: `${models.length} available`,
        gpu_memory: {
          used_gb: (usedGpuMem / 1024).toFixed(1),
          free_gb: ((totalGpuMem - usedGpuMem) / 1024).toFixed(1),
          total_gb: (totalGpuMem / 1024).toFixed(1),
        },
      },
      workers: workers.map((w: any) => ({
        id: w.id,
        name: w.name,
        status: w.status,
        gpus: (w.gpu_info || []).map((g: any) => ({
          index: g.index,
          name: g.name,
          memory_used_gb: (g.memory_used / 1024).toFixed(1),
          memory_free_gb: ((g.memory_total - g.memory_used) / 1024).toFixed(1),
          memory_total_gb: (g.memory_total / 1024).toFixed(1),
          utilization: g.utilization_gpu,
        })),
      })),
      running_deployments: runningDeployments.map((d: any) => ({
        id: d.id,
        model: d.model?.name || d.name,
        worker: d.worker?.name,
      })),
    },
    null,
    2,
  );
}

async function listWorkers(): Promise<string> {
  const response = await api.get("/workers");
  const workers = response.data.items || [];

  return JSON.stringify(
    workers.map((w: any) => ({
      id: w.id,
      name: w.name,
      host: w.host,
      status: w.status,
      gpus: (w.gpu_info || []).map((g: any) => ({
        index: g.index,
        name: g.name,
        memory_used_gb: (g.memory_used / 1024).toFixed(1),
        memory_free_gb: ((g.memory_total - g.memory_used) / 1024).toFixed(1),
        memory_total_gb: (g.memory_total / 1024).toFixed(1),
        utilization_percent: g.utilization_gpu,
      })),
    })),
    null,
    2,
  );
}

async function listContainers(
  status?: string,
  workerId?: number,
): Promise<string> {
  const response = await api.get("/containers");
  let containers = response.data.items || [];

  if (workerId) {
    containers = containers.filter(
      (c: any) => c.worker?.id === workerId || c.worker_id === workerId,
    );
  }

  if (status && status !== "all") {
    containers = containers.filter((c: any) => {
      const s = c.status?.toLowerCase() || "";
      if (status === "running") {
        return s.includes("running") || s.includes("up");
      }
      return s.includes(status);
    });
  }

  return JSON.stringify(
    containers.map((c: any) => ({
      id: c.id?.substring(0, 12),
      name: c.name,
      image: c.image,
      status: c.status,
      worker: c.worker?.name || c.worker_name,
      worker_id: c.worker?.id || c.worker_id,
    })),
    null,
    2,
  );
}

async function listDeployments(status?: string): Promise<string> {
  const response = await api.get("/deployments");
  let deployments = response.data.items || [];

  if (status && status !== "all") {
    deployments = deployments.filter((d: any) => d.status === status);
  }

  return JSON.stringify(
    deployments.map((d: any) => ({
      id: d.id,
      name: d.name,
      model: d.model?.name,
      model_id: d.model?.id,
      worker: d.worker?.name,
      worker_id: d.worker?.id,
      status: d.status,
      gpu_ids: d.gpu_ids,
      port: d.port,
      created_at: d.created_at,
    })),
    null,
    2,
  );
}

async function listModels(source?: string): Promise<string> {
  const response = await api.get("/models");
  let models = response.data.items || [];

  if (source) {
    models = models.filter((m: any) => m.source === source);
  }

  return JSON.stringify(
    models.map((m: any) => ({
      id: m.id,
      name: m.name,
      source: m.source,
      parameters: m.parameters,
      quantization: m.quantization,
    })),
    null,
    2,
  );
}

async function getGpuStatus(workerId?: number): Promise<string> {
  const response = await api.get("/workers");
  let workers = response.data.items || [];

  if (workerId) {
    workers = workers.filter((w: any) => w.id === workerId);
  }

  const result = workers.map((w: any) => ({
    worker_id: w.id,
    worker_name: w.name,
    status: w.status,
    gpus: (w.gpu_info || []).map((g: any) => ({
      index: g.index,
      name: g.name,
      memory_used_gb: (g.memory_used / 1024).toFixed(1),
      memory_free_gb: ((g.memory_total - g.memory_used) / 1024).toFixed(1),
      memory_total_gb: (g.memory_total / 1024).toFixed(1),
      utilization_percent: g.utilization_gpu,
      temperature: g.temperature,
    })),
  }));

  return JSON.stringify(result, null, 2);
}

async function deployModel(
  modelId: number,
  workerId: number,
  gpuIds?: number[],
): Promise<string> {
  if (!modelId || !workerId) {
    return "Error: model_id and worker_id are required";
  }

  const response = await api.post("/deployments", {
    model_id: modelId,
    worker_id: workerId,
    gpu_ids: gpuIds,
  });

  const deployment = response.data;
  const deploymentId = deployment.id;

  // Poll deployment status for up to 60 seconds
  const maxPollTime = 60000;
  const pollInterval = 3000;
  const startTime = Date.now();
  let lastStatus = deployment.status;
  const statusUpdates: string[] = [`Initial status: ${deployment.status}`];

  while (Date.now() - startTime < maxPollTime) {
    await new Promise((resolve) => setTimeout(resolve, pollInterval));

    try {
      const statusResponse = await api.get(`/deployments/${deploymentId}`);
      const currentStatus = statusResponse.data.status;
      const statusMessage = statusResponse.data.status_message;

      if (currentStatus !== lastStatus) {
        statusUpdates.push(
          `Status changed: ${lastStatus} → ${currentStatus}${statusMessage ? ` (${statusMessage})` : ""}`,
        );
        lastStatus = currentStatus;
      }

      // Stop polling if deployment reached a terminal state
      if (["running", "error", "stopped"].includes(currentStatus)) {
        return JSON.stringify(
          {
            success: currentStatus === "running",
            message:
              currentStatus === "running"
                ? `Deployment completed successfully! Model is now running.`
                : currentStatus === "error"
                  ? `Deployment failed: ${statusMessage || "Unknown error"}`
                  : `Deployment stopped`,
            deployment: {
              id: deploymentId,
              status: currentStatus,
              status_message: statusMessage,
              model: deployment.model?.name,
              worker: deployment.worker?.name,
              port: statusResponse.data.port,
            },
            status_history: statusUpdates,
          },
          null,
          2,
        );
      }
    } catch (error) {
      // Continue polling even if one request fails
    }
  }

  // Timeout - return current status
  return JSON.stringify(
    {
      success: false,
      message: `Deployment is still in progress (status: ${lastStatus}). Check [Deployments](/deployments) page for updates.`,
      deployment: {
        id: deploymentId,
        status: lastStatus,
        model: deployment.model?.name,
        worker: deployment.worker?.name,
      },
      status_history: statusUpdates,
      note: "Deployment is taking longer than expected. This is normal for large models that need to be downloaded.",
    },
    null,
    2,
  );
}

async function stopDeployment(deploymentId: number): Promise<string> {
  if (!deploymentId) {
    return "Error: deployment_id is required";
  }

  await api.post(`/deployments/${deploymentId}/stop`);
  return JSON.stringify(
    {
      success: true,
      message: `Deployment ${deploymentId} stopped successfully`,
    },
    null,
    2,
  );
}

async function startDeployment(deploymentId: number): Promise<string> {
  if (!deploymentId) {
    return "Error: deployment_id is required";
  }

  await api.post(`/deployments/${deploymentId}/start`);
  return JSON.stringify(
    {
      success: true,
      message: `Deployment ${deploymentId} started successfully`,
    },
    null,
    2,
  );
}

async function deleteDeployment(deploymentId: number): Promise<string> {
  if (!deploymentId) {
    return "Error: deployment_id is required";
  }

  await api.delete(`/deployments/${deploymentId}`);
  return JSON.stringify(
    {
      success: true,
      message: `Deployment ${deploymentId} deleted successfully`,
    },
    null,
    2,
  );
}

// ============================================================================
// Model Management Tools
// ============================================================================

async function addModel(
  name: string,
  source: string,
  parameters?: string,
  quantization?: string,
): Promise<string> {
  if (!name || !source) {
    return "Error: name and source are required";
  }

  const response = await api.post("/models", {
    name,
    source,
    parameters,
    quantization,
  });

  const model = response.data;
  return JSON.stringify(
    {
      success: true,
      message: `Model added successfully`,
      model: {
        id: model.id,
        name: model.name,
        source: model.source,
        parameters: model.parameters,
        quantization: model.quantization,
      },
    },
    null,
    2,
  );
}

async function deleteModel(modelId: number): Promise<string> {
  if (!modelId) {
    return "Error: model_id is required";
  }

  await api.delete(`/models/${modelId}`);
  return JSON.stringify(
    {
      success: true,
      message: `Model ${modelId} deleted successfully`,
    },
    null,
    2,
  );
}

// ============================================================================
// Container Tools
// ============================================================================

async function stopContainer(
  containerName: string,
  workerId: number,
): Promise<string> {
  if (!containerName || !workerId) {
    return "Error: container_name and worker_id are required";
  }

  // Backend expects: POST /containers/{container_id}/stop?worker_id=X
  await api.post(
    `/containers/${encodeURIComponent(containerName)}/stop`,
    null,
    {
      params: { worker_id: workerId },
    },
  );

  return JSON.stringify(
    {
      success: true,
      message: `Container "${containerName}" stopped successfully`,
    },
    null,
    2,
  );
}

async function removeContainer(
  containerName: string,
  workerId: number,
  force?: boolean,
): Promise<string> {
  if (!containerName || !workerId) {
    return "Error: container_name and worker_id are required";
  }

  // Backend expects: DELETE /containers/{container_id}?worker_id=X&force=Y
  await api.delete(`/containers/${encodeURIComponent(containerName)}`, {
    params: { worker_id: workerId, force: force || false },
  });

  return JSON.stringify(
    {
      success: true,
      message: `Container "${containerName}" removed successfully`,
    },
    null,
    2,
  );
}

// ============================================================================
// API Key Tools
// ============================================================================

async function listApiKeys(): Promise<string> {
  const response = await api.get("/api-keys");
  const apiKeys = response.data.items || [];

  return JSON.stringify(
    {
      total: response.data.total || apiKeys.length,
      api_keys: apiKeys.map((k: any) => ({
        id: k.id,
        name: k.name,
        description: k.description,
        access_key: k.access_key,
        expires_at: k.expires_at,
        created_at: k.created_at,
        last_used_at: k.last_used_at,
      })),
    },
    null,
    2,
  );
}

async function createApiKey(
  name: string,
  description?: string,
  expiresInDays?: number,
): Promise<string> {
  if (!name) {
    return "Error: name is required";
  }

  const response = await api.post("/api-keys", {
    name,
    description,
    expires_in_days: expiresInDays,
  });

  const apiKey = response.data;
  return JSON.stringify(
    {
      success: true,
      message: "API key created successfully",
      api_key: {
        id: apiKey.id,
        name: apiKey.name,
        access_key: apiKey.access_key,
        full_key: apiKey.api_key, // The full key is only shown once!
        expires_at: apiKey.expires_at,
      },
      warning: "Save the full API key now! It will not be shown again.",
    },
    null,
    2,
  );
}

async function deleteApiKey(apiKeyId: number): Promise<string> {
  if (!apiKeyId) {
    return "Error: api_key_id is required";
  }

  await api.delete(`/api-keys/${apiKeyId}`);
  return JSON.stringify(
    {
      success: true,
      message: `API key ${apiKeyId} deleted successfully`,
    },
    null,
    2,
  );
}

// ============================================================================
// Docker Image Tools
// ============================================================================

async function listImages(
  workerId?: number,
  repository?: string,
): Promise<string> {
  const params: any = {};
  if (workerId) params.worker_id = workerId;
  if (repository) params.repository = repository;

  const response = await api.get("/images", { params });
  const images = response.data.items || [];

  return JSON.stringify(
    {
      total: response.data.total || images.length,
      images: images.map((img: any) => ({
        id: img.id?.substring(0, 12),
        repository: img.repository,
        tag: img.tag,
        full_name: img.full_name,
        size_mb: (img.size / 1024 / 1024).toFixed(1),
        created_at: img.created_at,
        worker: img.worker_name,
        worker_id: img.worker_id,
      })),
    },
    null,
    2,
  );
}

async function pullImage(workerId: number, image: string): Promise<string> {
  if (!workerId || !image) {
    return "Error: worker_id and image are required";
  }

  const response = await api.post("/images/pull", {
    worker_id: workerId,
    image,
  });

  return JSON.stringify(
    {
      success: true,
      message: `Image "${image}" pulled successfully`,
      image: response.data.image,
    },
    null,
    2,
  );
}

async function deleteImage(
  imageId: string,
  workerId: number,
  force?: boolean,
): Promise<string> {
  if (!imageId || !workerId) {
    return "Error: image_id and worker_id are required";
  }

  await api.delete(`/images/${encodeURIComponent(imageId)}`, {
    params: { worker_id: workerId, force: force || false },
  });

  return JSON.stringify(
    {
      success: true,
      message: `Image "${imageId}" deleted successfully`,
    },
    null,
    2,
  );
}

// ============================================================================
// Storage Tools
// ============================================================================

async function listStorageVolumes(workerId?: number): Promise<string> {
  const params: any = {};
  if (workerId) params.worker_id = workerId;

  const response = await api.get("/storage/volumes", { params });
  const volumes = Array.isArray(response.data) ? response.data : [];

  return JSON.stringify(
    {
      total: volumes.length,
      volumes: volumes.map((v: any) => ({
        name: v.name,
        driver: v.driver,
        mountpoint: v.mountpoint,
        created_at: v.created_at,
        worker: v.worker_name,
        worker_id: v.worker_id,
      })),
    },
    null,
    2,
  );
}

async function getDiskUsage(workerId?: number): Promise<string> {
  const params: any = {};
  if (workerId) params.worker_id = workerId;

  const response = await api.get("/storage/disk-usage", { params });
  const usageList = Array.isArray(response.data) ? response.data : [];

  const formatSize = (bytes: number) => {
    if (bytes >= 1024 * 1024 * 1024) {
      return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  return JSON.stringify(
    {
      workers: usageList.map((u: any) => ({
        worker: u.worker_name,
        worker_id: u.worker_id,
        images: {
          count: u.images.count,
          size: formatSize(u.images.size),
          reclaimable: formatSize(u.images.reclaimable),
        },
        containers: {
          count: u.containers.count,
          size: formatSize(u.containers.size),
          reclaimable: formatSize(u.containers.reclaimable),
        },
        volumes: {
          count: u.volumes.count,
          size: formatSize(u.volumes.size),
          reclaimable: formatSize(u.volumes.reclaimable),
        },
        build_cache: {
          count: u.build_cache.count,
          size: formatSize(u.build_cache.size),
          reclaimable: formatSize(u.build_cache.reclaimable),
        },
        total_size: formatSize(u.total_size),
        total_reclaimable: formatSize(u.total_reclaimable),
      })),
    },
    null,
    2,
  );
}

async function deleteStorageVolume(
  volumeName: string,
  workerId: number,
  force?: boolean,
): Promise<string> {
  if (!volumeName || !workerId) {
    return "Error: volume_name and worker_id are required";
  }

  await api.delete(`/storage/volumes/${encodeURIComponent(volumeName)}`, {
    params: { worker_id: workerId, force: force || false },
  });

  return JSON.stringify(
    {
      success: true,
      message: `Volume "${volumeName}" deleted successfully`,
    },
    null,
    2,
  );
}

async function pruneStorage(
  workerId?: number,
  images: boolean = true,
  containers: boolean = true,
  volumes: boolean = false,
  buildCache: boolean = true,
): Promise<string> {
  const params: any = {};
  if (workerId) params.worker_id = workerId;

  const response = await api.post(
    "/storage/prune",
    {
      images,
      containers,
      volumes,
      build_cache: buildCache,
    },
    { params },
  );

  const results = Array.isArray(response.data) ? response.data : [];

  const formatSize = (bytes: number) => {
    if (bytes >= 1024 * 1024 * 1024) {
      return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  return JSON.stringify(
    {
      success: true,
      message: "Storage pruned successfully",
      results: results.map((r: any) => ({
        worker: r.worker_name,
        images_deleted: r.images_deleted,
        containers_deleted: r.containers_deleted,
        volumes_deleted: r.volumes_deleted,
        build_cache_deleted: r.build_cache_deleted,
        space_reclaimed: formatSize(r.space_reclaimed),
      })),
    },
    null,
    2,
  );
}

// ============================================================================
// Auto-Tuning Tools
// ============================================================================

async function listTuningJobs(status?: string): Promise<string> {
  const response = await api.get("/auto-tuning/jobs");
  let jobs = response.data.items || [];

  if (status) {
    jobs = jobs.filter((j: any) => j.status === status);
  }

  return JSON.stringify(
    {
      total: jobs.length,
      jobs: jobs.map((j: any) => ({
        id: j.id,
        model: j.model_name,
        worker: j.worker_name,
        optimization_target: j.optimization_target,
        status: j.status,
        progress: j.progress
          ? {
              step: j.progress.step,
              total_steps: j.progress.total_steps,
              step_name: j.progress.step_name,
              configs_tested: j.progress.configs_tested,
              configs_total: j.progress.configs_total,
              best_score: j.progress.best_score_so_far,
            }
          : null,
        created_at: j.created_at,
      })),
      note: "View detailed results at [Auto-Tuning](/auto-tuning) page",
    },
    null,
    2,
  );
}

async function startAutoTuning(
  modelId: number,
  workerId: number,
  optimizationTarget?: string,
  modelConfig?: ChatModelConfig,
): Promise<string> {
  if (!modelId || !workerId) {
    return "Error: model_id and worker_id are required";
  }

  // Build the LLM configuration for the agent
  const llmConfig: Record<string, any> = {};
  if (modelConfig) {
    if (modelConfig.type === "deployment" && modelConfig.deploymentId) {
      // Use local deployment
      llmConfig.deployment_id = modelConfig.deploymentId;
    } else if (modelConfig.type === "custom" && modelConfig.endpoint) {
      // Use custom endpoint
      llmConfig.base_url = modelConfig.endpoint;
      llmConfig.api_key = modelConfig.apiKey || "";
      llmConfig.model = modelConfig.modelId || modelConfig.name;
    }
  }

  const response = await api.post("/auto-tuning/jobs", {
    model_id: modelId,
    worker_id: workerId,
    optimization_target: optimizationTarget || "balanced",
    llm_config: Object.keys(llmConfig).length > 0 ? llmConfig : undefined,
  });

  const job = response.data;

  return JSON.stringify(
    {
      success: true,
      message:
        "Auto-tuning job started! The agent will use the selected model for reasoning.",
      job: {
        id: job.id,
        model: job.model_name,
        worker: job.worker_name,
        optimization_target: job.optimization_target,
        status: job.status,
        agent_llm: modelConfig?.name || "auto-detected",
      },
      note: "Track progress at [Auto-Tuning](/auto-tuning) page. This may take several minutes depending on the number of configurations to test.",
    },
    null,
    2,
  );
}

async function getTuningJob(jobId: number): Promise<string> {
  if (!jobId) {
    return "Error: job_id is required";
  }

  const response = await api.get(`/auto-tuning/jobs/${jobId}`);
  const job = response.data;

  return JSON.stringify(
    {
      id: job.id,
      model: job.model_name,
      worker: job.worker_name,
      optimization_target: job.optimization_target,
      status: job.status,
      status_message: job.status_message,
      progress: job.progress
        ? {
            step: job.progress.step,
            total_steps: job.progress.total_steps,
            step_name: job.progress.step_name,
            step_description: job.progress.step_description,
            configs_tested: job.progress.configs_tested,
            configs_total: job.progress.configs_total,
            current_config: job.progress.current_config,
            best_config_so_far: job.progress.best_config_so_far,
            best_score_so_far: job.progress.best_score_so_far,
          }
        : null,
      best_config: job.best_config,
      all_results: job.all_results,
      created_at: job.created_at,
      completed_at: job.completed_at,
    },
    null,
    2,
  );
}

async function cancelTuningJob(jobId: number): Promise<string> {
  if (!jobId) {
    return "Error: job_id is required";
  }

  await api.post(`/auto-tuning/jobs/${jobId}/cancel`);

  return JSON.stringify(
    {
      success: true,
      message: `Tuning job ${jobId} cancelled successfully`,
    },
    null,
    2,
  );
}

async function queryKnowledgeBase(
  modelName?: string,
  modelFamily?: string,
  gpuModel?: string,
  optimizationTarget?: string,
  limit?: number,
): Promise<string> {
  const response = await api.post("/auto-tuning/knowledge/query", {
    model_name: modelName,
    model_family: modelFamily,
    gpu_model: gpuModel,
    optimization_target: optimizationTarget || "balanced",
    limit: limit || 10,
  });

  const data = response.data;

  return JSON.stringify(
    {
      total: data.total,
      query: data.query,
      results: data.items.map((r: any) => ({
        gpu: `${r.gpu_count}x ${r.gpu_model}`,
        total_vram_gb: r.total_vram_gb,
        model: r.model_name,
        model_family: r.model_family,
        model_params_b: r.model_params_b,
        engine: r.engine,
        quantization: r.quantization,
        tensor_parallel: r.tensor_parallel,
        throughput_tps: r.throughput_tps,
        ttft_ms: r.ttft_ms,
        tpot_ms: r.tpot_ms,
        score: r.score,
      })),
      note: "These are benchmark results from similar configurations. Use this to guide your deployment decisions.",
    },
    null,
    2,
  );
}

async function runBenchmark(
  deploymentId: number,
  testType?: string,
  durationSeconds?: number,
  inputLength?: number,
  outputLength?: number,
  concurrency?: number,
): Promise<string> {
  if (!deploymentId) {
    return "Error: deployment_id is required";
  }

  const response = await api.post("/auto-tuning/benchmarks/run", {
    deployment_id: deploymentId,
    test_type: testType || "throughput",
    duration_seconds: durationSeconds || 60,
    input_length: inputLength || 512,
    output_length: outputLength || 128,
    concurrency: concurrency || 1,
  });

  const result = response.data;

  return JSON.stringify(
    {
      success: true,
      message: "Benchmark completed",
      benchmark: {
        id: result.id,
        deployment_id: result.deployment_id,
        test_type: result.test_type,
        duration_seconds: result.test_duration_seconds,
        config: {
          input_length: result.input_length,
          output_length: result.output_length,
          concurrency: result.concurrency,
        },
        metrics: {
          throughput_tps: result.metrics?.throughput_tps,
          ttft_ms: result.metrics?.ttft_ms,
          tpot_ms: result.metrics?.tpot_ms,
          total_latency_ms: result.metrics?.total_latency_ms,
          gpu_utilization: result.metrics?.gpu_utilization,
          vram_usage_gb: result.metrics?.vram_usage_gb,
        },
      },
      note: "Higher throughput (TPS) is better. Lower latency (TTFT, TPOT) is better.",
    },
    null,
    2,
  );
}
