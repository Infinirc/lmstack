/**
 * Application Constants
 *
 * Centralized configuration values used across the application.
 * Organized by category for maintainability.
 *
 * @module constants
 */

/**
 * Timing constants for polling and refresh intervals
 */
export const TIMING = {
  /** Default data refresh interval in milliseconds */
  REFRESH_INTERVAL: 10000,
  /** Workers page refresh interval */
  WORKERS_REFRESH: 5000,
  /** Dashboard refresh interval */
  DASHBOARD_REFRESH: 10000,
  /** Theme check interval */
  THEME_CHECK: 500,
} as const

/**
 * Pagination defaults
 */
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 10,
  PAGE_SIZE_OPTIONS: ['10', '20', '50', '100'],
} as const

/**
 * Local storage keys
 */
export const STORAGE_KEYS = {
  THEME: 'lmstack-theme',
  TOKEN: 'lmstack-token',
  USER: 'lmstack-user',
} as const

/**
 * Status color mappings for consistent styling
 */
export const STATUS_COLORS = {
  // Worker statuses
  online: 'green',
  offline: 'default',
  error: 'red',

  // Deployment statuses
  running: 'green',
  pending: 'blue',
  downloading: 'cyan',
  starting: 'blue',
  stopping: 'orange',
  stopped: 'default',

  // Model file statuses
  ready: 'green',
} as const

/**
 * Get status color with fallback
 */
export function getStatusColor(status: string): string {
  return STATUS_COLORS[status as keyof typeof STATUS_COLORS] || 'default'
}

/**
 * User role display names
 */
export const USER_ROLES = {
  admin: 'Administrator',
  operator: 'Operator',
  viewer: 'Viewer',
} as const

/**
 * Model backend options
 */
export const MODEL_BACKENDS = [
  { value: 'vllm', label: 'vLLM' },
  { value: 'ollama', label: 'Ollama' },
] as const

/**
 * Default card styles
 */
export const CARD_STYLES = {
  borderRadius: 12,
} as const

/**
 * Tag styles
 */
export const TAG_STYLES = {
  borderRadius: 6,
} as const

/**
 * API endpoints (relative to base URL)
 */
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    LOGOUT: '/auth/logout',
    ME: '/auth/me',
    SETUP: '/auth/setup',
    SETUP_STATUS: '/auth/setup/status',
    USERS: '/auth/users',
  },
  WORKERS: '/workers',
  MODELS: '/models',
  DEPLOYMENTS: '/deployments',
  MODEL_FILES: '/model-files',
  DASHBOARD: '/dashboard',
  API_KEYS: '/api-keys',
} as const
