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

export interface SystemInfo {
  cpu?: CPUInfo;
  memory?: MemoryInfo;
  disk?: DiskInfo;
}

export type ConnectionType = "direct" | "tailscale";

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
