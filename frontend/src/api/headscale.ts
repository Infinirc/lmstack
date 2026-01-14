/**
 * Headscale VPN API
 */
import { api } from './client'

export interface HeadscaleStatus {
  enabled: boolean
  running: boolean
  container_status?: string
  server_url?: string
  nodes_count?: number
  online_nodes?: number
}

export interface HeadscaleNode {
  id: number
  name: string
  given_name?: string
  ip_addresses: string[]
  ipv4?: string
  online: boolean
  last_seen?: string
  created_at?: string
}

export interface PreauthKeyResponse {
  key: string
  join_command: string
}

export const headscaleApi = {
  getStatus: async (): Promise<HeadscaleStatus> => {
    const response = await api.get<HeadscaleStatus>('/headscale/status')
    return response.data
  },

  start: async (data?: { server_url?: string; http_port?: number; grpc_port?: number }): Promise<HeadscaleStatus> => {
    const response = await api.post<HeadscaleStatus>('/headscale/start', data || {})
    return response.data
  },

  stop: async (): Promise<void> => {
    await api.post('/headscale/stop')
  },

  createPreauthKey: async (data?: {
    reusable?: boolean
    ephemeral?: boolean
    expiration?: string
    tags?: string[]
  }): Promise<PreauthKeyResponse> => {
    const response = await api.post<PreauthKeyResponse>('/headscale/preauth-key', data || {})
    return response.data
  },

  listNodes: async (): Promise<{ items: HeadscaleNode[]; total: number }> => {
    const response = await api.get<{ items: HeadscaleNode[]; total: number }>('/headscale/nodes')
    return response.data
  },

  deleteNode: async (nodeId: number): Promise<void> => {
    await api.delete(`/headscale/nodes/${nodeId}`)
  },
}
