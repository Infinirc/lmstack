/**
 * System API (Admin only)
 */
import { api } from './client'

export interface BackupInfo {
  filename: string
  size: number
  created_at: string
}

export interface BackupListResponse {
  items: BackupInfo[]
  total: number
}

export interface MessageResponse {
  message: string
  details?: string
}

export const systemApi = {
  clearStats: async (): Promise<MessageResponse> => {
    const response = await api.post<MessageResponse>('/system/clear-stats')
    return response.data
  },

  listBackups: async (): Promise<BackupListResponse> => {
    const response = await api.get<BackupListResponse>('/system/backups')
    return response.data
  },

  createBackup: async (): Promise<MessageResponse> => {
    const response = await api.post<MessageResponse>('/system/backup')
    return response.data
  },

  downloadBackup: (filename: string): string => {
    return `${api.defaults.baseURL}/system/backup/${filename}`
  },

  restoreBackup: async (filename: string): Promise<MessageResponse> => {
    const response = await api.post<MessageResponse>(`/system/restore/${filename}`)
    return response.data
  },

  deleteBackup: async (filename: string): Promise<MessageResponse> => {
    const response = await api.delete<MessageResponse>(`/system/backup/${filename}`)
    return response.data
  },

  restoreFromUpload: async (file: File): Promise<MessageResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post<MessageResponse>('/system/restore-upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },
}
