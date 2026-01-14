/**
 * Container and Image Types
 */

// Image Types
export interface ContainerImage {
  id: string;
  worker_id: number;
  worker_name: string;
  repository: string;
  tag: string;
  full_name: string;
  size: number;
  created_at: string;
  pulled_at?: string;
  digest?: string;
  labels?: Record<string, string>;
}

export interface ContainerImageDetail extends ContainerImage {
  layers: {
    digest: string;
    size: number;
    instruction?: string;
  }[];
  config: {
    env?: string[];
    cmd?: string[];
    entrypoint?: string[];
    working_dir?: string;
    exposed_ports?: string[];
    volumes?: string[];
  };
}

export interface ImagePullRequest {
  worker_id: number;
  image: string;
  registry_auth?: {
    username: string;
    password: string;
    server_address?: string;
  };
}

export interface ImageBuildRequest {
  worker_id: number;
  dockerfile: string;
  tag: string;
  build_args?: Record<string, string>;
}

export interface ImageOperationProgress {
  id: string;
  status: "pending" | "in_progress" | "complete" | "error";
  progress?: number;
  message?: string;
  current?: number;
  total?: number;
}

// Container Types
export type ContainerState =
  | "created"
  | "running"
  | "paused"
  | "restarting"
  | "removing"
  | "exited"
  | "dead";

export interface PortMapping {
  container_port: number;
  host_port: number;
  protocol: "tcp" | "udp";
  host_ip?: string;
}

export interface VolumeMount {
  source: string;
  destination: string;
  mode: "ro" | "rw";
  type: "bind" | "volume" | "tmpfs";
}

export interface ContainerStats {
  cpu_percent: number;
  memory_usage: number;
  memory_limit: number;
  memory_percent: number;
  network_rx: number;
  network_tx: number;
  block_read: number;
  block_write: number;
  pids: number;
}

export interface Container {
  id: string;
  worker_id: number;
  worker_name: string;
  name: string;
  image: string;
  image_id: string;
  state: ContainerState;
  status: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  exit_code?: number;
  ports: PortMapping[];
  volumes: VolumeMount[];
  env?: string[];
  gpu_ids?: string[];
  deployment_id?: number;
  deployment_name?: string;
  is_managed: boolean;
  stats?: ContainerStats;
}

export interface ContainerCreateRequest {
  worker_id: number;
  name: string;
  image: string;
  command?: string[];
  entrypoint?: string[];
  env?: Record<string, string>;
  ports?: {
    container_port: number;
    host_port?: number;
    protocol?: "tcp" | "udp";
  }[];
  volumes?: {
    source: string;
    destination: string;
    mode?: "ro" | "rw";
  }[];
  gpu_ids?: number[];
  restart_policy?: "no" | "always" | "on-failure" | "unless-stopped";
  labels?: Record<string, string>;
  cpu_limit?: number;
  memory_limit?: number;
}

export interface ContainerExecRequest {
  command: string[];
  tty?: boolean;
  privileged?: boolean;
  user?: string;
  workdir?: string;
  env?: string[];
}

export interface ContainerExecResult {
  exit_code: number;
  stdout: string;
  stderr: string;
}

export interface ContainerLogsResponse {
  container_id: string;
  logs: string;
  stdout?: string;
  stderr?: string;
}
