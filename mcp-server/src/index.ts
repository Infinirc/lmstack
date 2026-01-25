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
            gpu_ids: {
              type: "array",
              items: { type: "number" },
              description: "GPU indices to use (optional, defaults to auto-select)",
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
          args.gpu_ids as number[] | undefined
        );

        return {
          content: [
            {
              type: "text",
              text: `Successfully started deployment!\n\nDeployment ID: ${result.id}\nStatus: ${result.status}\n\nThe model is being deployed. Use list_deployments to check status.`,
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
