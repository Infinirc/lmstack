/**
 * User and Auth Types
 */

export type UserRole = 'admin' | 'operator' | 'viewer'

export interface User {
  id: number
  username: string
  email?: string
  display_name?: string
  role: UserRole
  is_active: boolean
  created_at: string
  last_login_at?: string
}

export interface UserCreate {
  username: string
  password: string
  email?: string
  display_name?: string
  role?: string
}

export interface UserUpdate {
  email?: string
  display_name?: string
  role?: string
  is_active?: boolean
}

export interface LoginRequest {
  username: string
  password: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  user: User
}

export interface SetupRequest {
  username: string
  password: string
  email?: string
}

export interface SetupStatus {
  initialized: boolean
}
