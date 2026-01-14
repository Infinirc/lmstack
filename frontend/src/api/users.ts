/**
 * Users API (Admin only)
 */
import { api } from './client'
import type { User, UserCreate, UserUpdate, ListResponse } from '../types'

export interface UserListParams {
  skip?: number
  limit?: number
}

export const usersApi = {
  list: async (params?: UserListParams): Promise<ListResponse<User>> => {
    const response = await api.get<ListResponse<User>>('/auth/users', { params })
    return response.data
  },

  get: async (id: number): Promise<User> => {
    const response = await api.get<User>(`/auth/users/${id}`)
    return response.data
  },

  create: async (data: UserCreate): Promise<User> => {
    const response = await api.post<User>('/auth/users', data)
    return response.data
  },

  update: async (id: number, data: UserUpdate): Promise<User> => {
    const response = await api.patch<User>(`/auth/users/${id}`, data)
    return response.data
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/auth/users/${id}`)
  },
}
