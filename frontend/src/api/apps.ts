/**
 * Deploy Apps API
 */
import { api } from './client'
import type { ListResponse } from '../types'

export interface AppDefinition {
  type: string
  name: string
  description: string
  image: string
}

export interface DeployedApp {
  id: number
  app_type: string
  name: string
  worker_id: number
  worker_name?: string
  worker_address?: string
  status: string
  status_message?: string
  container_id?: string
  port?: number
  proxy_path: string
  proxy_url?: string
  use_proxy: boolean
  access_url?: string
  api_key_id?: number
  created_at: string
  updated_at: string
}

export interface AppDeployRequest {
  app_type: string
  worker_id: number
  name?: string
  use_proxy?: boolean
}

export interface DeployProgress {
  stage: string
  progress: number
  message: string
}

export interface AppLogsResponse {
  app_id: number
  logs: string
}

export const appsApi = {
  listAvailable: async (): Promise<{ items: AppDefinition[] }> => {
    const response = await api.get<{ items: AppDefinition[] }>('/apps/available')
    return response.data
  },

  getProgress: async (id: number): Promise<DeployProgress> => {
    const response = await api.get<DeployProgress>(`/apps/${id}/progress`)
    return response.data
  },

  list: async (): Promise<ListResponse<DeployedApp>> => {
    const response = await api.get<ListResponse<DeployedApp>>('/apps')
    return response.data
  },

  get: async (id: number): Promise<DeployedApp> => {
    const response = await api.get<DeployedApp>(`/apps/${id}`)
    return response.data
  },

  deploy: async (data: AppDeployRequest): Promise<DeployedApp> => {
    const response = await api.post<DeployedApp>('/apps', data)
    return response.data
  },

  stop: async (id: number): Promise<DeployedApp> => {
    const response = await api.post<DeployedApp>(`/apps/${id}/stop`)
    return response.data
  },

  start: async (id: number): Promise<DeployedApp> => {
    const response = await api.post<DeployedApp>(`/apps/${id}/start`)
    return response.data
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/apps/${id}`)
  },

  getLogs: async (id: number, tail: number = 500): Promise<AppLogsResponse> => {
    const response = await api.get<AppLogsResponse>(`/apps/${id}/logs`, {
      params: { tail },
    })
    return response.data
  },
}
