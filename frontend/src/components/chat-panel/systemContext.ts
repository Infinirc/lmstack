/**
 * System Context Builder
 *
 * Builds system context for the AI assistant to understand the current
 * state of the LMStack platform.
 */
import { api } from "../../api/client";

export interface SystemContext {
  workers: WorkerInfo[];
  deployments: DeploymentInfo[];
  models: ModelInfo[];
  containers: ContainerInfo[];
  images: ImageInfo[];
  storageVolumes: StorageVolumeInfo[];
  semanticRouter: SemanticRouterInfo | null;
  timestamp: string;
}

interface WorkerInfo {
  id: number;
  name: string;
  host: string;
  status: string;
  gpus: GpuInfo[];
}

interface GpuInfo {
  index: number;
  name: string;
  memoryTotal: number;
  memoryUsed: number;
  utilizationGpu: number;
}

interface DeploymentInfo {
  id: number;
  name: string;
  modelName: string;
  workerName: string;
  status: string;
  endpoint?: string;
}

interface ModelInfo {
  id: number;
  name: string;
  source: string;
  parameters?: string;
  quantization?: string;
}

interface ContainerInfo {
  id: string;
  name: string;
  image: string;
  status: string;
  workerName: string;
}

interface ImageInfo {
  id: string;
  name: string;
  tag: string;
  size: number;
  workerName: string;
}

interface StorageVolumeInfo {
  name: string;
  driver: string;
  mountpoint: string;
  workerName: string;
}

interface SemanticRouterInfo {
  deployed: boolean;
  status: string;
  models: string[];
}

/**
 * Fetch current system state
 */
export async function fetchSystemContext(): Promise<SystemContext> {
  try {
    const [
      workersRes,
      deploymentsRes,
      modelsRes,
      containersRes,
      imagesRes,
      storageRes,
      srStatus,
    ] = await Promise.all([
      api.get("/workers").catch(() => ({ data: { items: [] } })),
      api.get("/deployments").catch(() => ({ data: { items: [] } })),
      api.get("/models").catch(() => ({ data: { items: [] } })),
      api.get("/containers").catch(() => ({ data: { items: [] } })),
      api.get("/images").catch(() => ({ data: { items: [] } })),
      api.get("/storage/volumes").catch(() => ({ data: [] })),
      api.get("/semantic-router/status").catch(() => ({ data: null })),
    ]);

    const workers: WorkerInfo[] = (workersRes.data.items || []).map(
      (w: any) => ({
        id: w.id,
        name: w.name,
        host: w.host,
        status: w.status,
        gpus: (w.gpu_info || []).map((g: any) => ({
          index: g.index,
          name: g.name,
          memoryTotal: g.memory_total,
          memoryUsed: g.memory_used,
          utilizationGpu: g.utilization_gpu,
        })),
      }),
    );

    const deployments: DeploymentInfo[] = (deploymentsRes.data.items || []).map(
      (d: any) => ({
        id: d.id,
        name: d.name,
        modelName: d.model?.name || d.name,
        workerName: d.worker?.name || "unknown",
        status: d.status,
        endpoint:
          d.status === "running" ? `/api/deployments/${d.id}/chat` : undefined,
      }),
    );

    const models: ModelInfo[] = (modelsRes.data.items || []).map((m: any) => ({
      id: m.id,
      name: m.name,
      source: m.source,
      parameters: m.parameters,
      quantization: m.quantization,
    }));

    const containers: ContainerInfo[] = (containersRes.data.items || []).map(
      (c: any) => ({
        id: c.id || c.container_id,
        name: c.name,
        image: c.image,
        status: c.status,
        workerName: c.worker?.name || c.worker_name || "unknown",
      }),
    );

    const images: ImageInfo[] = (imagesRes.data.items || []).map((i: any) => ({
      id: i.id || i.image_id,
      name: i.name || i.repository,
      tag: i.tag || "latest",
      size: i.size || 0,
      workerName: i.worker?.name || i.worker_name || "unknown",
    }));

    // Backend /storage/volumes returns a list directly, not { items: [] }
    const storageVolumes: StorageVolumeInfo[] = (
      Array.isArray(storageRes.data) ? storageRes.data : []
    ).map((v: any) => ({
      name: v.name,
      driver: v.driver || "local",
      mountpoint: v.mountpoint || "",
      workerName: v.worker_name || "unknown",
    }));

    const semanticRouter: SemanticRouterInfo | null = srStatus.data
      ? {
          deployed: srStatus.data.deployed,
          status: srStatus.data.status,
          models: srStatus.data.models || [],
        }
      : null;

    return {
      workers,
      deployments,
      models,
      containers,
      images,
      storageVolumes,
      semanticRouter,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    console.error("Failed to fetch system context:", error);
    return {
      workers: [],
      deployments: [],
      models: [],
      containers: [],
      images: [],
      storageVolumes: [],
      semanticRouter: null,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Page routes for navigation links
 */
const PAGE_ROUTES = {
  dashboard: "/dashboard",
  workers: "/workers",
  containers: "/containers",
  images: "/images",
  storage: "/storage",
  models: "/models",
  deployments: "/deployments",
  chat: "/chat",
  apiKeys: "/api-keys",
  settings: "/settings",
};

/**
 * Action links that open modals directly
 */
const ACTION_LINKS = {
  newDeployment: "/deployments?action=new",
  newModel: "/models?action=new",
  newApiKey: "/api-keys?action=new",
};

/**
 * Format system context as a system message for the LLM
 */
export function formatSystemPrompt(context: SystemContext): string {
  const lines: string[] = [
    "You are an AI assistant for the LMStack platform - an LLM deployment and management system.",
    "You have access to real-time system information and can help users manage their AI infrastructure.",
    "",
    "## Navigation Links",
    "When referencing pages, use markdown links so users can click to navigate:",
    `- Workers: [Workers](${PAGE_ROUTES.workers})`,
    `- Docker Containers: [Containers](${PAGE_ROUTES.containers})`,
    `- Docker Images: [Images](${PAGE_ROUTES.images})`,
    `- Storage Volumes: [Storage](${PAGE_ROUTES.storage})`,
    `- Models: [Models](${PAGE_ROUTES.models})`,
    `- Deployments: [Deployments](${PAGE_ROUTES.deployments})`,
    `- API Keys: [API Keys](${PAGE_ROUTES.apiKeys})`,
    "",
    "## Quick Action Links (ALWAYS use for create/deploy/add operations)",
    "**CRITICAL: For deploying models, adding models, or creating API keys, ALWAYS guide users to the UI. NEVER use deploy_model, add_model, or create_api_key tools directly.**",
    "",
    "These links open the action dialog directly:",
    `- Deploy a Model: [New Deployment](${ACTION_LINKS.newDeployment})`,
    `- Add a Model: [Add Model](${ACTION_LINKS.newModel})`,
    `- Create API Key: [Create API Key](${ACTION_LINKS.newApiKey})`,
    "",
    "**TOOL USAGE RULES:**",
    "- deploy_model, add_model, create_api_key → NEVER use these. Always guide to UI instead.",
    "- stop_deployment, delete_deployment, stop_container, remove_container, delete_* → OK to use (destructive actions need confirmation)",
    "- list_*, get_* → OK to use (query tools, no confirmation needed)",
    "",
    "**EXAMPLES:**",
    `- User: '我想部署模型' → '請點擊 [New Deployment](${ACTION_LINKS.newDeployment}) 開啟部署表單。'`,
    `- User: '幫我部署 Qwen' → '請點擊 [New Deployment](${ACTION_LINKS.newDeployment}) 來部署，選擇 Qwen 模型即可。'`,
    `- User: '幫我新增模型' → '請點擊 [Add Model](${ACTION_LINKS.newModel}) 來新增模型。'`,
    "- User: '有哪些模型?' → Use list_models tool",
    "- User: '停止 deployment 1' → Use stop_deployment tool",
    "",
    "## Current System Status",
    `Last updated: ${new Date(context.timestamp).toLocaleString()}`,
    "",
  ];

  // Workers section
  lines.push("### Workers");
  if (context.workers.length === 0) {
    lines.push("No workers registered. Go to [Workers](/workers) to add one.");
  } else {
    lines.push(`Total: ${context.workers.length} worker(s)`);
    for (const worker of context.workers) {
      lines.push(`- **${worker.name}** (${worker.host}): ${worker.status}`);
      for (const gpu of worker.gpus) {
        const memUsedGB = (gpu.memoryUsed / 1024).toFixed(1);
        const memTotalGB = (gpu.memoryTotal / 1024).toFixed(1);
        const memFreeGB = ((gpu.memoryTotal - gpu.memoryUsed) / 1024).toFixed(
          1,
        );
        lines.push(
          `  - GPU ${gpu.index}: ${gpu.name}, Used: ${memUsedGB}GB, Free: ${memFreeGB}GB, Total: ${memTotalGB}GB, Util: ${gpu.utilizationGpu}%`,
        );
      }
    }
  }
  lines.push("");

  // Docker Containers section
  lines.push("### Docker Containers");
  if (context.containers.length === 0) {
    lines.push(
      "No containers running. View [Containers](/containers) page for details.",
    );
  } else {
    const runningContainers = context.containers.filter(
      (c) =>
        c.status.toLowerCase().includes("running") ||
        c.status.toLowerCase().includes("up"),
    );
    lines.push(
      `Total: ${context.containers.length} container(s), Running: ${runningContainers.length}`,
    );
    for (const container of context.containers.slice(0, 10)) {
      lines.push(
        `- **${container.name}** (${container.image}): ${container.status} on ${container.workerName}`,
      );
    }
    if (context.containers.length > 10) {
      lines.push(
        `  ... and ${context.containers.length - 10} more. See [Containers](/containers) for full list.`,
      );
    }
  }
  lines.push("");

  // Docker Images section
  lines.push("### Docker Images");
  if (context.images.length === 0) {
    lines.push("No images found. View [Images](/images) page for details.");
  } else {
    lines.push(`Total: ${context.images.length} image(s)`);
    const imagesByWorker: Record<string, typeof context.images> = {};
    for (const img of context.images) {
      if (!imagesByWorker[img.workerName]) imagesByWorker[img.workerName] = [];
      imagesByWorker[img.workerName].push(img);
    }
    for (const [worker, imgs] of Object.entries(imagesByWorker)) {
      lines.push(`- ${worker}: ${imgs.length} image(s)`);
    }
  }
  lines.push("");

  // Storage section
  lines.push("### Storage Volumes");
  if (context.storageVolumes.length === 0) {
    lines.push(
      "No storage volumes. View [Storage](/storage) page for details.",
    );
  } else {
    lines.push(`Total: ${context.storageVolumes.length} volume(s)`);
  }
  lines.push("");

  // Deployments section
  lines.push("### Model Deployments");
  const activeDeployments = context.deployments.filter(
    (d) => d.status === "running",
  );
  const allDeployments = context.deployments;
  lines.push(
    `Total: ${allDeployments.length}, Running: ${activeDeployments.length}`,
  );
  if (activeDeployments.length === 0) {
    lines.push(
      "No active model deployments. Go to [Deployments](/deployments) to deploy a model.",
    );
  } else {
    for (const dep of activeDeployments) {
      lines.push(
        `- **${dep.modelName}** on ${dep.workerName} (ID: ${dep.id}) - running`,
      );
    }
  }
  lines.push("");

  // Available models section
  lines.push("### Available Models");
  if (context.models.length === 0) {
    lines.push("No models registered. Go to [Models](/models) to add models.");
  } else {
    lines.push(`Total: ${context.models.length} model(s)`);
    const modelsBySource: Record<string, typeof context.models> = {};
    for (const model of context.models) {
      if (!modelsBySource[model.source]) {
        modelsBySource[model.source] = [];
      }
      modelsBySource[model.source].push(model);
    }
    for (const [source, models] of Object.entries(modelsBySource)) {
      lines.push(`- **${source}:** ${models.map((m) => m.name).join(", ")}`);
    }
  }
  lines.push("");

  // Semantic Router
  if (context.semanticRouter) {
    lines.push("### Semantic Router");
    lines.push(
      `Status: ${context.semanticRouter.deployed ? "Deployed" : "Not deployed"}`,
    );
    if (context.semanticRouter.models.length > 0) {
      lines.push(
        `Connected models: ${context.semanticRouter.models.join(", ")}`,
      );
    }
    lines.push("");
  }

  // Tool Calling Capabilities
  lines.push("## Available Actions (Tool Calling)");
  lines.push(
    "You have access to tools that allow you to TAKE ACTIONS on the system:",
  );
  lines.push("");
  lines.push("### Query Tools (No confirmation needed)");
  lines.push("- `get_system_status`: Get complete system overview");
  lines.push("- `list_workers`: List all workers with GPU information");
  lines.push(
    "- `list_containers`: List Docker containers (filter by status/worker_id)",
  );
  lines.push("- `list_deployments`: List model deployments (filter by status)");
  lines.push("- `list_models`: List available models (filter by source)");
  lines.push("- `get_gpu_status`: Get detailed GPU status");
  lines.push("");
  lines.push("### Model Management (Requires user confirmation)");
  lines.push(
    "- `add_model`: Add a new model (name, source: huggingface/ollama)",
  );
  lines.push("- `delete_model`: Delete a model (model_id)");
  lines.push("");
  lines.push("### Deployment Management (Requires user confirmation)");
  lines.push(
    "- `deploy_model`: Deploy a model to a worker (model_id, worker_id, gpu_ids?)",
  );
  lines.push("- `stop_deployment`: Stop a running deployment (deployment_id)");
  lines.push(
    "- `start_deployment`: Start a stopped deployment (deployment_id)",
  );
  lines.push(
    "- `delete_deployment`: Delete a deployment permanently (deployment_id)",
  );
  lines.push("");
  lines.push("### Container Management (Requires user confirmation)");
  lines.push(
    "- `stop_container`: Stop a Docker container (container_name, worker_id)",
  );
  lines.push(
    "- `remove_container`: Remove a Docker container (container_name, worker_id, force?)",
  );
  lines.push("");
  lines.push("### API Key Management");
  lines.push("- `list_api_keys`: List all API keys (No confirmation needed)");
  lines.push(
    "- `create_api_key`: Create a new API key (name, description?, expires_in_days?)",
  );
  lines.push("- `delete_api_key`: Delete an API key (api_key_id)");
  lines.push("");
  lines.push("### Docker Image Management");
  lines.push(
    "- `list_images`: List all Docker images (No confirmation needed)",
  );
  lines.push("- `pull_image`: Pull a Docker image (worker_id, image)");
  lines.push(
    "- `delete_image`: Delete a Docker image (image_id, worker_id, force?)",
  );
  lines.push("");
  lines.push("### Storage Management");
  lines.push(
    "- `list_storage_volumes`: List storage volumes (No confirmation needed)",
  );
  lines.push(
    "- `get_disk_usage`: Get disk usage statistics (No confirmation needed)",
  );
  lines.push(
    "- `delete_storage_volume`: Delete a storage volume (volume_name, worker_id, force?)",
  );
  lines.push(
    "- `prune_storage`: Clean up unused Docker resources (images?, containers?, volumes?, build_cache?)",
  );
  lines.push("");
  lines.push("**WORKFLOW FOR CONTAINER/IMAGE OPERATIONS:**");
  lines.push(
    "1. If user asks to stop/remove a container by name, FIRST call list_containers to find the worker_id",
  );
  lines.push(
    "2. Then call stop_container or remove_container with both container_name AND worker_id",
  );
  lines.push(
    "3. Same workflow applies to images - use list_images first to find worker_id",
  );
  lines.push("");
  lines.push(
    "**IMPORTANT:** When users ask you to perform actions, USE THE TOOLS to execute them. The user will see a confirmation dialog before any action is executed.",
  );
  lines.push("");

  // Capabilities and instructions
  lines.push("## Instructions");
  lines.push(
    "1. Always use markdown links when mentioning pages (e.g., [Containers](/containers))",
  );
  lines.push("2. Provide specific numbers from the system data above");
  lines.push("3. Be concise and accurate");
  lines.push(
    "4. When users ask about 'containers' or 'docker containers', refer to the Docker Containers section",
  );
  lines.push(
    "5. When users ask about 'deployments', distinguish between Docker containers and Model Deployments",
  );
  lines.push(
    "6. When users ask you to deploy a model, USE the deploy_model tool",
  );
  lines.push(
    "7. When users ask you to stop/start/delete a deployment, USE the corresponding tool",
  );
  lines.push("8. After executing an action, report the result to the user");
  lines.push("9. Respond in the same language as the user's query");
  lines.push("");

  return lines.join("\n");
}
