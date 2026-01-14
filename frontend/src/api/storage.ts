/**
 * Storage API
 */
import { api } from './client'

export interface StorageCategory {
  count: number
  size: number
  reclaimable: number
}

export interface DiskUsage {
  worker_id: number
  worker_name: string
  images: StorageCategory
  containers: StorageCategory
  volumes: StorageCategory
  build_cache: StorageCategory
  total_size: number
  total_reclaimable: number
}

export interface Volume {
  name: string
  worker_id: number
  worker_name: string
  driver: string
  mountpoint: string
  created_at: string
  labels: Record<string, string>
  scope: string
}

export interface PruneRequest {
  images?: boolean
  containers?: boolean
  volumes?: boolean
  build_cache?: boolean
}

export interface PruneResult {
  worker_id: number
  worker_name: string
  images_deleted: number
  containers_deleted: number
  volumes_deleted: number
  build_cache_deleted: number
  space_reclaimed: number
}

export const storageApi = {
  getDiskUsage: async (workerId?: number): Promise<DiskUsage[]> => {
    const response = await api.get<DiskUsage[]>('/storage/disk-usage', {
      params: workerId ? { worker_id: workerId } : undefined,
    })
    return response.data
  },

  listVolumes: async (workerId?: number): Promise<Volume[]> => {
    const response = await api.get<Volume[]>('/storage/volumes', {
      params: workerId ? { worker_id: workerId } : undefined,
    })
    return response.data
  },

  deleteVolume: async (volumeName: string, workerId: number, force?: boolean): Promise<void> => {
    await api.delete(`/storage/volumes/${volumeName}`, {
      params: { worker_id: workerId, force },
    })
  },

  prune: async (options?: PruneRequest, workerId?: number): Promise<PruneResult[]> => {
    const response = await api.post<PruneResult[]>('/storage/prune', options || {}, {
      params: workerId ? { worker_id: workerId } : undefined,
    })
    return response.data
  },
}
