/**
 * Dashboard Types
 */

export interface ResourceCounts {
  worker_count: number;
  worker_online_count: number;
  gpu_count: number;
  model_count: number;
  deployment_count: number;
  deployment_running_count: number;
}

export interface GPUSummary {
  total_memory_gb: number;
  used_memory_gb: number;
  utilization_avg: number;
  temperature_avg: number;
  temperature_max: number;
}

export interface UsagePoint {
  date: string;
  value: number;
}

export interface UsageSummary {
  total_requests: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  request_history: UsagePoint[];
  token_history: UsagePoint[];
}

export interface TopModel {
  model_id: number;
  model_name: string;
  request_count: number;
  token_count: number;
}

export interface TopApiKey {
  api_key_id: number;
  api_key_name: string;
  request_count: number;
  token_count: number;
}

export interface DashboardData {
  resources: ResourceCounts;
  gpu_summary: GPUSummary;
  usage: UsageSummary;
  top_models: TopModel[];
  top_api_keys: TopApiKey[];
}
