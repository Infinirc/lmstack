/**
 * Deployment Types
 */
import type { ModelBackend } from './model'

export type DeploymentStatus =
  | 'pending'
  | 'downloading'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'error'

export interface WorkerSummary {
  id: number
  name: string
  address: string
  status: string
}

export interface ModelSummary {
  id: number
  name: string
  model_id: string
  source: string
}

export interface Deployment {
  id: number
  name: string
  model_id: number
  worker_id: number
  backend: ModelBackend
  status: DeploymentStatus
  status_message?: string
  container_id?: string
  port?: number
  gpu_indexes?: number[]
  extra_params?: Record<string, unknown>
  created_at: string
  updated_at: string
  worker?: WorkerSummary
  model?: ModelSummary
}

export interface DeploymentCreate {
  name: string
  model_id: number
  worker_id: number
  backend: ModelBackend
  gpu_indexes?: number[]
  extra_params?: Record<string, unknown>
}
