/**
 * Model Types
 */

export type ModelBackend = 'vllm' | 'ollama' | 'sglang'
export type ModelSource = 'huggingface' | 'ollama'

export interface LLMModel {
  id: number
  name: string
  model_id: string
  source: ModelSource
  description?: string
  default_params?: Record<string, unknown>
  docker_image?: string
  created_at: string
  updated_at: string
  deployment_count: number
}

export interface LLMModelCreate {
  name: string
  model_id: string
  source?: ModelSource
  description?: string
  default_params?: Record<string, unknown>
  docker_image?: string
}

export interface ModelFileDeployment {
  id: number
  name: string
  status: string
  port?: number
}

export type ModelFileStatus = 'downloading' | 'starting' | 'ready' | 'stopped'

export interface ModelFileView {
  model_id: number
  worker_id: number
  model_name: string
  model_source: string
  worker_name: string
  worker_address: string
  status: ModelFileStatus
  download_progress: number
  deployment_count: number
  running_count: number
  deployments: ModelFileDeployment[]
}
