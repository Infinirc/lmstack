/**
 * Error Handling Utilities
 *
 * Provides consistent error handling across the application.
 *
 * @module utils/error
 */

import { message } from 'antd'

/**
 * API error response structure
 */
interface ApiErrorResponse {
  response?: {
    data?: {
      detail?: string
      message?: string
    }
    status?: number
    statusText?: string
  }
  message?: string
}

/**
 * Extract error message from various error types
 */
export function getErrorMessage(error: unknown, fallback = 'An error occurred'): string {
  if (!error) return fallback

  // Handle API error response
  const apiError = error as ApiErrorResponse
  if (apiError.response?.data?.detail) {
    return apiError.response.data.detail
  }
  if (apiError.response?.data?.message) {
    return apiError.response.data.message
  }

  // Handle Error object
  if (error instanceof Error) {
    return error.message
  }

  // Handle string error
  if (typeof error === 'string') {
    return error
  }

  return fallback
}

/**
 * Show error message toast
 */
export function showError(error: unknown, fallback?: string): void {
  message.error(getErrorMessage(error, fallback))
}

/**
 * Show success message toast
 */
export function showSuccess(msg: string): void {
  message.success(msg)
}

/**
 * Show info message toast
 */
export function showInfo(msg: string): void {
  message.info(msg)
}

/**
 * Log error to console with context
 */
export function logError(context: string, error: unknown): void {
  console.error(`[${context}]`, error)
}

/**
 * Handle API error with logging and toast
 */
export function handleApiError(context: string, error: unknown, fallback?: string): void {
  logError(context, error)
  showError(error, fallback || `Failed to ${context.toLowerCase()}`)
}
