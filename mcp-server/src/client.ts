/**
 * LMStack API Client
 *
 * HTTP client for communicating with LMStack backend API.
 * This client is shared between MCP Server and Web Chat tools.
 */
import axios, { AxiosInstance } from "axios";

export class LMStackClient {
  private client: AxiosInstance;

  constructor(baseURL: string, token?: string) {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    });
  }

  // ============================================================================
  // Workers
  // ============================================================================

  async getWorkers(): Promise<any[]> {
    const response = await this.client.get("/workers");
    return response.data.items || [];
  }

  // ============================================================================
  // Containers
  // ============================================================================

  async getContainers(): Promise<any[]> {
    const response = await this.client.get("/containers");
    return response.data.items || [];
  }

  async stopContainer(containerName: string, workerId: number): Promise<void> {
    await this.client.post(`/containers/${encodeURIComponent(containerName)}/stop`, null, {
      params: { worker_id: workerId },
    });
  }

  async removeContainer(containerName: string, workerId: number, force?: boolean): Promise<void> {
    await this.client.delete(`/containers/${encodeURIComponent(containerName)}`, {
      params: { worker_id: workerId, force: force || false },
    });
  }

  // ============================================================================
  // Models
  // ============================================================================

  async getModels(): Promise<any[]> {
    const response = await this.client.get("/models");
    return response.data.items || [];
  }

  async addModel(
    name: string,
    source: string,
    parameters?: string,
    quantization?: string
  ): Promise<any> {
    const response = await this.client.post("/models", {
      name,
      source,
      parameters,
      quantization,
    });
    return response.data;
  }

  async deleteModel(modelId: number): Promise<void> {
    await this.client.delete(`/models/${modelId}`);
  }

  // ============================================================================
  // Deployments
  // ============================================================================

  async getDeployments(): Promise<any[]> {
    const response = await this.client.get("/deployments");
    return response.data.items || [];
  }

  async getDeployment(deploymentId: number): Promise<any> {
    const response = await this.client.get(`/deployments/${deploymentId}`);
    return response.data;
  }

  async deployModel(
    modelId: number,
    workerId: number,
    gpuIds?: number[]
  ): Promise<any> {
    const response = await this.client.post("/deployments", {
      model_id: modelId,
      worker_id: workerId,
      gpu_ids: gpuIds,
    });
    return response.data;
  }

  async stopDeployment(deploymentId: number): Promise<void> {
    await this.client.post(`/deployments/${deploymentId}/stop`);
  }

  async startDeployment(deploymentId: number): Promise<void> {
    await this.client.post(`/deployments/${deploymentId}/start`);
  }

  async deleteDeployment(deploymentId: number): Promise<void> {
    await this.client.delete(`/deployments/${deploymentId}`);
  }

  // ============================================================================
  // API Keys
  // ============================================================================

  async getApiKeys(): Promise<any> {
    const response = await this.client.get("/api-keys");
    return response.data;
  }

  async createApiKey(
    name: string,
    description?: string,
    expiresInDays?: number
  ): Promise<any> {
    const response = await this.client.post("/api-keys", {
      name,
      description,
      expires_in_days: expiresInDays,
    });
    return response.data;
  }

  async deleteApiKey(apiKeyId: number): Promise<void> {
    await this.client.delete(`/api-keys/${apiKeyId}`);
  }

  // ============================================================================
  // Docker Images
  // ============================================================================

  async getImages(workerId?: number, repository?: string): Promise<any[]> {
    const params: any = {};
    if (workerId) params.worker_id = workerId;
    if (repository) params.repository = repository;

    const response = await this.client.get("/images", { params });
    return response.data.items || [];
  }

  async pullImage(workerId: number, image: string): Promise<any> {
    const response = await this.client.post("/images/pull", {
      worker_id: workerId,
      image,
    });
    return response.data;
  }

  async deleteImage(imageId: string, workerId: number, force?: boolean): Promise<void> {
    await this.client.delete(`/images/${encodeURIComponent(imageId)}`, {
      params: { worker_id: workerId, force: force || false },
    });
  }

  // ============================================================================
  // Storage
  // ============================================================================

  async getStorageVolumes(workerId?: number): Promise<any[]> {
    const params: any = {};
    if (workerId) params.worker_id = workerId;

    const response = await this.client.get("/storage/volumes", { params });
    return Array.isArray(response.data) ? response.data : [];
  }

  async getDiskUsage(workerId?: number): Promise<any[]> {
    const params: any = {};
    if (workerId) params.worker_id = workerId;

    const response = await this.client.get("/storage/disk-usage", { params });
    return Array.isArray(response.data) ? response.data : [];
  }

  async deleteStorageVolume(volumeName: string, workerId: number, force?: boolean): Promise<void> {
    await this.client.delete(`/storage/volumes/${encodeURIComponent(volumeName)}`, {
      params: { worker_id: workerId, force: force || false },
    });
  }

  async pruneStorage(
    workerId?: number,
    images: boolean = true,
    containers: boolean = true,
    volumes: boolean = false,
    buildCache: boolean = true
  ): Promise<any[]> {
    const params: any = {};
    if (workerId) params.worker_id = workerId;

    const response = await this.client.post("/storage/prune", {
      images,
      containers,
      volumes,
      build_cache: buildCache,
    }, { params });

    return Array.isArray(response.data) ? response.data : [];
  }

  // ============================================================================
  // Dashboard
  // ============================================================================

  async getDashboard(): Promise<any> {
    const response = await this.client.get("/dashboard");
    return response.data;
  }
}
