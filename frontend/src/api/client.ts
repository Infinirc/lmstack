/**
 * API Client
 *
 * Configured axios instance with authentication and error handling.
 */
import axios, { type AxiosInstance, type AxiosError } from 'axios'
import { STORAGE_KEYS } from '../constants'

/**
 * API error response structure
 */
export interface ApiErrorResponse {
  detail?: string
  message?: string
}

/**
 * Create and configure axios instance
 */
function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: '/api',
    headers: {
      'Content-Type': 'application/json',
    },
  })

  // Request interceptor: Add auth token
  client.interceptors.request.use((config) => {
    const token = localStorage.getItem(STORAGE_KEYS.TOKEN)
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  })

  // Response interceptor: Handle 401 errors
  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError<ApiErrorResponse>) => {
      if (error.response?.status === 401) {
        localStorage.removeItem(STORAGE_KEYS.TOKEN)
        localStorage.removeItem(STORAGE_KEYS.USER)

        const isAuthPage =
          window.location.pathname.includes('/login') ||
          window.location.pathname.includes('/setup')

        if (!isAuthPage) {
          window.location.href = '/login'
        }
      }
      return Promise.reject(error)
    }
  )

  return client
}

export const api = createApiClient()
export default api
