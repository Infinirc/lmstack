/**
 * Worker Types
 */

export interface GPUInfo {
  index: number;
  name: string;
  memory_total: number;
  memory_used: number;
  memory_free: number;
  utilization: number;
  temperature?: number;
}

export interface CPUInfo {
  percent: number;
  count: number;
  freq_mhz: number;
}

export interface MemoryInfo {
  total: number;
  used: number;
  free: number;
  percent: number;
}

export interface DiskInfo {
  total: number;
  used: number;
  free: number;
  percent: number;
}

export interface CapabilitiesInfo {
  os_type: string;
  gpu_type: string;
  docker: boolean;
  ollama: boolean;
  ollama_running: boolean;
  mlx?: boolean;
  llama_cpp?: boolean;
}

export interface SystemInfo {
  cpu?: CPUInfo;
  memory?: MemoryInfo;
  disk?: DiskInfo;
  os_type?: string;
  gpu_type?: string;
  capabilities?: CapabilitiesInfo;
}

export type ConnectionType = "direct" | "tailscale";
export type OSType = "linux" | "darwin" | "windows";
export type GPUType = "nvidia" | "apple_silicon" | "amd" | "none";

export interface Worker {
  id: number;
  name: string;
  address: string;
  status: "online" | "offline" | "error";
  connection_type: ConnectionType;
  tailscale_ip?: string;
  headscale_node_id?: number;
  effective_address?: string;
  description?: string;
  labels?: Record<string, string>;
  gpu_info?: GPUInfo[];
  system_info?: SystemInfo;
  os_type: OSType;
  gpu_type: GPUType;
  capabilities?: CapabilitiesInfo;
  available_backends?: string[];
  created_at: string;
  updated_at: string;
  last_heartbeat?: string;
  deployment_count: number;
}

export interface WorkerCreate {
  name: string;
  address: string;
  description?: string;
  labels?: Record<string, string>;
  connection_type?: ConnectionType;
  tailscale_ip?: string;
  headscale_node_id?: number;
}

export interface RegistrationToken {
  id: number;
  token: string;
  name: string;
  is_used: boolean;
  used_by_worker_id?: number;
  created_at: string;
  expires_at: string;
  used_at?: string;
  is_valid: boolean;
  docker_command?: string;
}

export interface RegistrationTokenCreate {
  name: string;
  expires_in_hours?: number;
}

export interface LocalWorkerSpawnResponse {
  success: boolean;
  message: string;
  worker_name: string;
  container_id?: string;
  backend_url: string;
}
