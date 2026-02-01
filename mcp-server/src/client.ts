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
    name?: string,
    gpuIndexes?: number[],
    backend?: string
  ): Promise<any> {
    const response = await this.client.post("/deployments", {
      model_id: modelId,
      worker_id: workerId,
      name: name || `deployment-${modelId}-${Date.now()}`,
      gpu_indexes: gpuIndexes || [0],
      backend: backend || "vllm",
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

  // ============================================================================
  // Benchmark & Auto-Tuning
  // ============================================================================

  async runBenchmark(config: {
    deployment_id: number;
    test_type?: string;
    duration_seconds?: number;
    input_length?: number;
    output_length?: number;
    concurrency?: number;
  }): Promise<any> {
    try {
      const response = await this.client.post("/auto-tuning/benchmark", config);
      return response.data;
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  async createTuningJob(config: {
    model_id: number;
    worker_id: number;
    optimization_target?: string;
    tuning_config?: any;
    llm_config?: {
      deployment_id?: number;
      base_url?: string;
      api_key?: string;
      model?: string;
    };
  }): Promise<any> {
    try {
      const response = await this.client.post("/auto-tuning/jobs", config);
      return response.data;
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  async getTuningJob(jobId: number): Promise<any> {
    try {
      const response = await this.client.get(`/auto-tuning/jobs/${jobId}`);
      return response.data;
    } catch (error: any) {
      return null;
    }
  }

  async listTuningJobs(): Promise<any[]> {
    try {
      const response = await this.client.get("/auto-tuning/jobs");
      return response.data.items || [];
    } catch (error: any) {
      return [];
    }
  }

  async cancelTuningJob(jobId: number): Promise<boolean> {
    try {
      await this.client.post(`/auto-tuning/jobs/${jobId}/cancel`);
      return true;
    } catch (error: any) {
      return false;
    }
  }

  async queryKnowledgeBase(query: {
    model_name?: string;
    gpu_model?: string;
    limit?: number;
  }): Promise<any[]> {
    try {
      const response = await this.client.post("/auto-tuning/knowledge/query", query);
      return response.data.items || [];
    } catch (error: any) {
      return [];
    }
  }

  // ============================================================================
  // Comprehensive Benchmark
  // ============================================================================

  async runComprehensiveBenchmark(config: {
    deployment_id: number;
    concurrency?: number;
    num_requests?: number;
    warmup_requests?: number;
    prompt_tokens?: number;
    output_tokens?: number;
    custom_prompt?: string;
  }): Promise<any> {
    try {
      const response = await this.client.post("/auto-tuning/benchmarks/comprehensive", config, {
        timeout: 300000, // 5 minutes timeout for comprehensive benchmark
      });
      return response.data;
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  async runSaturationDetection(config: {
    deployment_id: number;
    start_concurrency?: number;
    max_concurrency?: number;
    requests_per_level?: number;
    use_exponential?: boolean;
    step_size?: number;
    step_multiplier?: number;
  }): Promise<any> {
    try {
      const response = await this.client.post("/auto-tuning/benchmarks/saturation", config, {
        timeout: 600000, // 10 minutes timeout for saturation detection
      });
      return response.data;
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }
}
