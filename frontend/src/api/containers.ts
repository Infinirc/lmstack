/**
 * Containers API
 */
import { api } from './client'
import type {
  Container,
  ContainerState,
  ContainerCreateRequest,
  ContainerExecRequest,
  ContainerExecResult,
  ContainerLogsResponse,
  ListResponse,
} from '../types'

export interface ContainerListParams {
  skip?: number
  limit?: number
  worker_id?: number
  state?: ContainerState
  all?: boolean
  managed_only?: boolean
}

export const containersApi = {
  list: async (params?: ContainerListParams): Promise<ListResponse<Container>> => {
    const response = await api.get<ListResponse<Container>>('/containers', { params })
    return response.data
  },

  get: async (containerId: string, workerId: number): Promise<Container> => {
    const response = await api.get<Container>(
      `/containers/${containerId}`,
      { params: { worker_id: workerId } }
    )
    return response.data
  },

  create: async (data: ContainerCreateRequest): Promise<Container> => {
    const response = await api.post<Container>('/containers', data)
    return response.data
  },

  start: async (containerId: string, workerId: number): Promise<Container> => {
    const response = await api.post<Container>(
      `/containers/${containerId}/start`,
      null,
      { params: { worker_id: workerId } }
    )
    return response.data
  },

  stop: async (containerId: string, workerId: number, timeout?: number): Promise<Container> => {
    const response = await api.post<Container>(
      `/containers/${containerId}/stop`,
      null,
      { params: { worker_id: workerId, timeout } }
    )
    return response.data
  },

  restart: async (containerId: string, workerId: number, timeout?: number): Promise<Container> => {
    const response = await api.post<Container>(
      `/containers/${containerId}/restart`,
      null,
      { params: { worker_id: workerId, timeout } }
    )
    return response.data
  },

  delete: async (containerId: string, workerId: number, force?: boolean, volumes?: boolean): Promise<void> => {
    await api.delete(`/containers/${containerId}`, {
      params: { worker_id: workerId, force, v: volumes },
    })
  },

  getLogs: async (
    containerId: string,
    workerId: number,
    options?: { tail?: number; since?: string; until?: string; timestamps?: boolean }
  ): Promise<ContainerLogsResponse> => {
    const response = await api.get<ContainerLogsResponse>(
      `/containers/${containerId}/logs`,
      { params: { worker_id: workerId, ...options } }
    )
    return response.data
  },

  exec: async (
    containerId: string,
    workerId: number,
    data: ContainerExecRequest
  ): Promise<ContainerExecResult> => {
    const response = await api.post<ContainerExecResult>(
      `/containers/${containerId}/exec`,
      data,
      { params: { worker_id: workerId } }
    )
    return response.data
  },

  pause: async (containerId: string, workerId: number): Promise<Container> => {
    const response = await api.post<Container>(
      `/containers/${containerId}/pause`,
      null,
      { params: { worker_id: workerId } }
    )
    return response.data
  },

  unpause: async (containerId: string, workerId: number): Promise<Container> => {
    const response = await api.post<Container>(
      `/containers/${containerId}/unpause`,
      null,
      { params: { worker_id: workerId } }
    )
    return response.data
  },
}
