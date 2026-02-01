/**
 * Data Formatters
 *
 * Format LMStack API responses into human-readable text for MCP.
 * These formatters produce markdown output suitable for AI agents.
 */

/**
 * Helper to format byte sizes
 */
function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }
  return `${(bytes / 1024).toFixed(2)} KB`;
}

/**
 * Format workers list
 */
export function formatWorkers(workers: any[]): string {
  if (workers.length === 0) {
    return "No workers registered.";
  }

  const lines: string[] = [`# Workers (${workers.length} total)\n`];

  for (const worker of workers) {
    const statusText = worker.status === "online" ? "[online]" : "[offline]";
    lines.push(`## ${worker.name} ${statusText}`);
    lines.push(`- **Host:** ${worker.host}`);
    lines.push(`- **Status:** ${worker.status}`);
    lines.push(`- **ID:** ${worker.id}`);

    if (worker.gpu_info && worker.gpu_info.length > 0) {
      lines.push(`- **GPUs:** ${worker.gpu_info.length}`);
      for (const gpu of worker.gpu_info) {
        // Memory values from pynvml are in bytes
        const bytesToGB = 1024 * 1024 * 1024;
        const usedGB = (gpu.memory_used / bytesToGB).toFixed(1);
        const totalGB = (gpu.memory_total / bytesToGB).toFixed(1);
        const freeGB = ((gpu.memory_total - gpu.memory_used) / bytesToGB).toFixed(1);
        const util = gpu.utilization_gpu ?? gpu.utilization ?? 0;
        lines.push(`  - GPU ${gpu.index}: ${gpu.name}`);
        lines.push(`    - Memory: ${usedGB}GB / ${totalGB}GB (${freeGB}GB free)`);
        lines.push(`    - Utilization: ${util}%`);
      }
    } else {
      lines.push(`- **GPUs:** None detected`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format containers list
 */
export function formatContainers(containers: any[]): string {
  if (containers.length === 0) {
    return "No containers found.";
  }

  const running = containers.filter(
    (c) => c.status?.toLowerCase().includes("running") || c.status?.toLowerCase().includes("up")
  );

  const lines: string[] = [
    `# Docker Containers (${containers.length} total, ${running.length} running)\n`,
  ];

  // Group by worker
  const byWorker: Record<string, any[]> = {};
  for (const container of containers) {
    const workerName = container.worker?.name || container.worker_name || "Unknown";
    if (!byWorker[workerName]) {
      byWorker[workerName] = [];
    }
    byWorker[workerName].push(container);
  }

  for (const [workerName, workerContainers] of Object.entries(byWorker)) {
    lines.push(`## ${workerName}`);
    for (const container of workerContainers) {
      const statusText = container.status?.toLowerCase().includes("running") ||
                          container.status?.toLowerCase().includes("up") ? "[running]" : "[stopped]";
      lines.push(`- **${container.name}** ${statusText}`);
      lines.push(`  - Image: ${container.image}`);
      lines.push(`  - Status: ${container.status}`);
      lines.push(`  - ID: ${container.id?.substring(0, 12) || "N/A"}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format deployments list
 */
export function formatDeployments(deployments: any[]): string {
  if (deployments.length === 0) {
    return "No model deployments found.";
  }

  const running = deployments.filter((d) => d.status === "running");

  const lines: string[] = [
    `# Model Deployments (${deployments.length} total, ${running.length} running)\n`,
  ];

  for (const dep of deployments) {
    const statusText =
      dep.status === "running" ? "[running]" :
      dep.status === "starting" ? "[starting]" :
      dep.status === "stopped" ? "[stopped]" : "[error]";

    lines.push(`## ${dep.model?.name || dep.name} ${statusText}`);
    lines.push(`- **ID:** ${dep.id}`);
    lines.push(`- **Status:** ${dep.status}`);
    lines.push(`- **Worker:** ${dep.worker?.name || "Unknown"}`);

    if (dep.status === "running" && dep.port) {
      lines.push(`- **Endpoint:** http://${dep.worker?.host}:${dep.port}/v1`);
    }

    if (dep.gpu_ids && dep.gpu_ids.length > 0) {
      lines.push(`- **GPUs:** ${dep.gpu_ids.join(", ")}`);
    }

    if (dep.created_at) {
      lines.push(`- **Created:** ${new Date(dep.created_at).toLocaleString()}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format models list
 */
export function formatModels(models: any[]): string {
  if (models.length === 0) {
    return "No models registered.";
  }

  const lines: string[] = [`# Available Models (${models.length} total)\n`];

  // Group by source
  const bySource: Record<string, any[]> = {};
  for (const model of models) {
    const source = model.source || "unknown";
    if (!bySource[source]) {
      bySource[source] = [];
    }
    bySource[source].push(model);
  }

  for (const [source, sourceModels] of Object.entries(bySource)) {
    lines.push(`## ${source.charAt(0).toUpperCase() + source.slice(1)} (${sourceModels.length})`);
    for (const model of sourceModels) {
      lines.push(`- **${model.name}** (ID: ${model.id})`);
      if (model.parameters) {
        lines.push(`  - Parameters: ${model.parameters}`);
      }
      if (model.quantization) {
        lines.push(`  - Quantization: ${model.quantization}`);
      }
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format complete system status
 */
export function formatSystemStatus(
  workers: any[],
  containers: any[],
  deployments: any[],
  models: any[]
): string {
  const onlineWorkers = workers.filter((w) => w.status === "online");
  const runningContainers = containers.filter(
    (c) => c.status?.toLowerCase().includes("running") || c.status?.toLowerCase().includes("up")
  );
  const runningDeployments = deployments.filter((d) => d.status === "running");

  // Calculate total GPU memory (values from pynvml are in bytes)
  const bytesToGB = 1024 * 1024 * 1024;
  let totalGpuMemory = 0;
  let usedGpuMemory = 0;
  for (const worker of workers) {
    for (const gpu of worker.gpu_info || []) {
      totalGpuMemory += gpu.memory_total || 0;
      usedGpuMemory += gpu.memory_used || 0;
    }
  }
  const freeGpuMemory = totalGpuMemory - usedGpuMemory;

  const lines: string[] = [
    "# LMStack System Status",
    "",
    `**Last Updated:** ${new Date().toLocaleString()}`,
    "",
    "## Summary",
    `- **Workers:** ${onlineWorkers.length}/${workers.length} online`,
    `- **Containers:** ${runningContainers.length}/${containers.length} running`,
    `- **Deployments:** ${runningDeployments.length}/${deployments.length} running`,
    `- **Models:** ${models.length} available`,
    `- **GPU Memory:** ${(usedGpuMemory / bytesToGB).toFixed(1)}GB used / ${(freeGpuMemory / bytesToGB).toFixed(1)}GB free / ${(totalGpuMemory / bytesToGB).toFixed(1)}GB total`,
    "",
  ];

  // Add workers section
  lines.push(formatWorkers(workers));
  lines.push("");

  // Add running deployments
  if (runningDeployments.length > 0) {
    lines.push("## Active Deployments");
    for (const dep of runningDeployments) {
      lines.push(`- **${dep.model?.name || dep.name}** on ${dep.worker?.name} (ID: ${dep.id})`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format API keys list
 */
export function formatApiKeys(apiKeysData: any): string {
  const apiKeys = apiKeysData?.items || [];

  if (apiKeys.length === 0) {
    return "No API keys found.";
  }

  const lines: string[] = [`# API Keys (${apiKeys.length} total)\n`];

  for (const key of apiKeys) {
    lines.push(`## ${key.name}`);
    lines.push(`- **ID:** ${key.id}`);
    lines.push(`- **Access Key:** ${key.access_key || "N/A"}`);
    if (key.description) {
      lines.push(`- **Description:** ${key.description}`);
    }
    if (key.expires_at) {
      lines.push(`- **Expires:** ${new Date(key.expires_at).toLocaleString()}`);
    } else {
      lines.push(`- **Expires:** Never`);
    }
    if (key.last_used_at) {
      lines.push(`- **Last Used:** ${new Date(key.last_used_at).toLocaleString()}`);
    }
    lines.push(`- **Created:** ${new Date(key.created_at).toLocaleString()}`);
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format Docker images list
 */
export function formatImages(images: any[]): string {
  if (images.length === 0) {
    return "No Docker images found.";
  }

  const lines: string[] = [`# Docker Images (${images.length} total)\n`];

  // Group by worker
  const byWorker: Record<string, any[]> = {};
  for (const img of images) {
    const workerName = img.worker_name || "Unknown";
    if (!byWorker[workerName]) {
      byWorker[workerName] = [];
    }
    byWorker[workerName].push(img);
  }

  for (const [workerName, workerImages] of Object.entries(byWorker)) {
    lines.push(`## ${workerName} (${workerImages.length} images)`);
    for (const img of workerImages) {
      const name = img.full_name || `${img.repository || ""}:${img.tag || "latest"}`;
      lines.push(`- **${name}**`);
      lines.push(`  - ID: ${img.id?.substring(0, 12) || "N/A"}`);
      lines.push(`  - Size: ${formatSize(img.size || 0)}`);
      if (img.created_at) {
        lines.push(`  - Created: ${new Date(img.created_at).toLocaleString()}`);
      }
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format storage volumes list
 */
export function formatStorageVolumes(volumes: any[]): string {
  if (volumes.length === 0) {
    return "No storage volumes found.";
  }

  const lines: string[] = [`# Storage Volumes (${volumes.length} total)\n`];

  // Group by worker
  const byWorker: Record<string, any[]> = {};
  for (const vol of volumes) {
    const workerName = vol.worker_name || "Unknown";
    if (!byWorker[workerName]) {
      byWorker[workerName] = [];
    }
    byWorker[workerName].push(vol);
  }

  for (const [workerName, workerVolumes] of Object.entries(byWorker)) {
    lines.push(`## ${workerName} (${workerVolumes.length} volumes)`);
    for (const vol of workerVolumes) {
      lines.push(`- **${vol.name}**`);
      lines.push(`  - Driver: ${vol.driver || "local"}`);
      if (vol.mountpoint) {
        lines.push(`  - Mountpoint: ${vol.mountpoint}`);
      }
      if (vol.created_at) {
        lines.push(`  - Created: ${new Date(vol.created_at).toLocaleString()}`);
      }
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format disk usage statistics
 */
export function formatDiskUsage(usageList: any[]): string {
  if (usageList.length === 0) {
    return "No disk usage data available.";
  }

  const lines: string[] = ["# Disk Usage\n"];

  for (const u of usageList) {
    lines.push(`## ${u.worker_name || "Worker"}`);

    lines.push("### Images");
    lines.push(`- Count: ${u.images?.count || 0}`);
    lines.push(`- Size: ${formatSize(u.images?.size || 0)}`);
    lines.push(`- Reclaimable: ${formatSize(u.images?.reclaimable || 0)}`);

    lines.push("### Containers");
    lines.push(`- Count: ${u.containers?.count || 0}`);
    lines.push(`- Size: ${formatSize(u.containers?.size || 0)}`);
    lines.push(`- Reclaimable: ${formatSize(u.containers?.reclaimable || 0)}`);

    lines.push("### Volumes");
    lines.push(`- Count: ${u.volumes?.count || 0}`);
    lines.push(`- Size: ${formatSize(u.volumes?.size || 0)}`);
    lines.push(`- Reclaimable: ${formatSize(u.volumes?.reclaimable || 0)}`);

    lines.push("### Build Cache");
    lines.push(`- Count: ${u.build_cache?.count || 0}`);
    lines.push(`- Size: ${formatSize(u.build_cache?.size || 0)}`);
    lines.push(`- Reclaimable: ${formatSize(u.build_cache?.reclaimable || 0)}`);

    lines.push("");
    lines.push(`**Total Size:** ${formatSize(u.total_size || 0)}`);
    lines.push(`**Total Reclaimable:** ${formatSize(u.total_reclaimable || 0)}`);
    lines.push("");
  }

  return lines.join("\n");
}
