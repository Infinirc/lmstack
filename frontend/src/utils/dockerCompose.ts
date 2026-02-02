/**
 * Docker Compose YAML generator and parser for LMStack deployments
 */

import YAML from "yaml";

export interface DeploymentConfig {
  name: string;
  model_id: string;
  model_name?: string;
  backend: string;
  worker_name?: string;
  gpu_indexes?: number[];
  extra_params?: {
    docker_image?: string;
    tensor_parallel_size?: number;
    max_model_len?: number;
    gpu_memory_utilization?: number;
    quantization?: string;
    dtype?: string;
    enforce_eager?: boolean;
    trust_remote_code?: boolean;
    [key: string]: unknown;
  };
}

interface DockerComposeService {
  image: string;
  container_name?: string;
  ports?: string[];
  environment?: Record<string, string>;
  volumes?: string[];
  deploy?: {
    resources?: {
      reservations?: {
        devices?: Array<{
          driver: string;
          count?: number | string;
          device_ids?: string[];
          capabilities: string[][];
        }>;
      };
    };
  };
  command?: string;
  restart?: string;
  shm_size?: string;
  ipc?: string;
}

interface DockerCompose {
  version: string;
  services: Record<string, DockerComposeService>;
}

// Default images for each backend
const DEFAULT_IMAGES: Record<string, string> = {
  vllm: "vllm/vllm-openai:latest",
  sglang: "lmsysorg/sglang:latest",
  ollama: "ollama/ollama:latest",
};

/**
 * Generate Docker Compose YAML from deployment config
 */
export function generateDockerCompose(config: DeploymentConfig): string {
  const backend = config.backend;
  const serviceName = config.name.replace(/[^a-zA-Z0-9_-]/g, "-").toLowerCase();

  const service: DockerComposeService = {
    image:
      config.extra_params?.docker_image ||
      DEFAULT_IMAGES[backend] ||
      "vllm/vllm-openai:latest",
    container_name: `lmstack-${serviceName}`,
    restart: "unless-stopped",
  };

  // GPU configuration
  if (config.gpu_indexes && config.gpu_indexes.length > 0) {
    service.deploy = {
      resources: {
        reservations: {
          devices: [
            {
              driver: "nvidia",
              device_ids: config.gpu_indexes.map(String),
              capabilities: [["gpu"]],
            },
          ],
        },
      },
    };
  } else {
    service.deploy = {
      resources: {
        reservations: {
          devices: [
            {
              driver: "nvidia",
              count: "all",
              capabilities: [["gpu"]],
            },
          ],
        },
      },
    };
  }

  // Backend-specific configuration
  if (backend === "vllm") {
    service.ports = ["8000:8000"];
    service.ipc = "host";

    const cmdParts = [`--model ${config.model_id}`];

    if (config.extra_params?.tensor_parallel_size) {
      cmdParts.push(
        `--tensor-parallel-size ${config.extra_params.tensor_parallel_size}`,
      );
    }
    if (config.extra_params?.max_model_len) {
      cmdParts.push(`--max-model-len ${config.extra_params.max_model_len}`);
    }
    if (config.extra_params?.gpu_memory_utilization) {
      cmdParts.push(
        `--gpu-memory-utilization ${config.extra_params.gpu_memory_utilization}`,
      );
    }
    if (config.extra_params?.quantization) {
      cmdParts.push(`--quantization ${config.extra_params.quantization}`);
    }
    if (config.extra_params?.dtype) {
      cmdParts.push(`--dtype ${config.extra_params.dtype}`);
    }
    if (config.extra_params?.enforce_eager) {
      cmdParts.push("--enforce-eager");
    }
    if (config.extra_params?.trust_remote_code) {
      cmdParts.push("--trust-remote-code");
    }

    service.command = cmdParts.join(" \\\n      ");
  } else if (backend === "sglang") {
    service.ports = ["30000:30000"];
    service.shm_size = "32g";

    const cmdParts = [
      "python3 -m sglang.launch_server",
      `--model-path ${config.model_id}`,
      "--host 0.0.0.0",
      "--port 30000",
    ];

    if (config.extra_params?.tensor_parallel_size) {
      cmdParts.push(`--tp ${config.extra_params.tensor_parallel_size}`);
    }
    if (config.extra_params?.max_model_len) {
      cmdParts.push(`--context-length ${config.extra_params.max_model_len}`);
    }
    if (config.extra_params?.quantization) {
      cmdParts.push(`--quantization ${config.extra_params.quantization}`);
    }
    if (config.extra_params?.trust_remote_code) {
      cmdParts.push("--trust-remote-code");
    }

    service.command = cmdParts.join(" \\\n      ");
  } else if (backend === "ollama") {
    service.ports = ["11434:11434"];
    service.volumes = ["ollama_data:/root/.ollama"];
    service.environment = {
      OLLAMA_HOST: "0.0.0.0:11434",
    };
  }

  const compose: DockerCompose = {
    version: "3.8",
    services: {
      [serviceName]: service,
    },
  };

  // Add volumes for Ollama
  if (backend === "ollama") {
    return YAML.stringify({
      ...compose,
      volumes: {
        ollama_data: {},
      },
    });
  }

  return YAML.stringify(compose);
}

/**
 * Parse Docker Compose YAML and extract deployment config
 */
export function parseDockerCompose(
  yamlContent: string,
): Partial<DeploymentConfig> | null {
  try {
    const compose = YAML.parse(yamlContent) as DockerCompose;

    if (!compose.services) {
      return null;
    }

    const serviceName = Object.keys(compose.services)[0];
    const service = compose.services[serviceName];

    if (!service) {
      return null;
    }

    const config: Partial<DeploymentConfig> = {
      name: serviceName,
      extra_params: {},
    };

    // Extract image
    if (service.image) {
      config.extra_params!.docker_image = service.image;

      // Detect backend from image
      if (service.image.includes("vllm")) {
        config.backend = "vllm";
      } else if (service.image.includes("sglang")) {
        config.backend = "sglang";
      } else if (service.image.includes("ollama")) {
        config.backend = "ollama";
      }
    }

    // Extract GPU indexes
    const devices = service.deploy?.resources?.reservations?.devices;
    if (devices && devices[0]?.device_ids) {
      config.gpu_indexes = devices[0].device_ids.map(Number);
    }

    // Parse command to extract params
    if (service.command) {
      const cmd = service.command;

      // Extract model
      const modelMatch = cmd.match(/--model(?:-path)?\s+(\S+)/);
      if (modelMatch) {
        config.model_id = modelMatch[1];
      }

      // Extract tensor_parallel_size
      const tpMatch = cmd.match(/--tensor-parallel-size\s+(\d+)|--tp\s+(\d+)/);
      if (tpMatch) {
        config.extra_params!.tensor_parallel_size = parseInt(
          tpMatch[1] || tpMatch[2],
        );
      }

      // Extract max_model_len
      const maxLenMatch = cmd.match(
        /--max-model-len\s+(\d+)|--context-length\s+(\d+)/,
      );
      if (maxLenMatch) {
        config.extra_params!.max_model_len = parseInt(
          maxLenMatch[1] || maxLenMatch[2],
        );
      }

      // Extract gpu_memory_utilization
      const gpuMemMatch = cmd.match(/--gpu-memory-utilization\s+([\d.]+)/);
      if (gpuMemMatch) {
        config.extra_params!.gpu_memory_utilization = parseFloat(
          gpuMemMatch[1],
        );
      }

      // Extract quantization
      const quantMatch = cmd.match(/--quantization\s+(\S+)/);
      if (quantMatch) {
        config.extra_params!.quantization = quantMatch[1];
      }

      // Extract dtype
      const dtypeMatch = cmd.match(/--dtype\s+(\S+)/);
      if (dtypeMatch) {
        config.extra_params!.dtype = dtypeMatch[1];
      }

      // Check for flags
      if (cmd.includes("--enforce-eager")) {
        config.extra_params!.enforce_eager = true;
      }
      if (cmd.includes("--trust-remote-code")) {
        config.extra_params!.trust_remote_code = true;
      }
    }

    return config;
  } catch (e) {
    console.error("Failed to parse Docker Compose YAML:", e);
    return null;
  }
}

/**
 * Validate Docker Compose YAML syntax
 */
export function validateDockerCompose(yamlContent: string): {
  valid: boolean;
  error?: string;
} {
  try {
    const parsed = YAML.parse(yamlContent);

    if (!parsed) {
      return { valid: false, error: "Empty YAML" };
    }

    if (!parsed.services) {
      return { valid: false, error: "Missing 'services' section" };
    }

    const serviceNames = Object.keys(parsed.services);
    if (serviceNames.length === 0) {
      return { valid: false, error: "No services defined" };
    }

    return { valid: true };
  } catch (e) {
    return {
      valid: false,
      error: `YAML syntax error: ${(e as Error).message}`,
    };
  }
}
