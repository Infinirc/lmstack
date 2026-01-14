/**
 * Common Types
 */

export interface ListResponse<T> {
  items: T[]
  total: number
}

// K8s Future Extension Types (placeholder)
export interface K8sCluster {
  id: number
  name: string
  api_server: string
  version?: string
  status: 'connected' | 'disconnected' | 'error'
  node_count: number
  created_at: string
}

export interface K8sNode {
  name: string
  cluster_id: number
  status: 'Ready' | 'NotReady' | 'Unknown'
  role: 'master' | 'worker'
  cpu_capacity: string
  memory_capacity: string
  gpu_count: number
  pods_count: number
}
