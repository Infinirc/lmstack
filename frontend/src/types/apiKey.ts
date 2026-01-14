/**
 * API Key Types
 */

export interface ApiKey {
  id: number;
  name: string;
  description?: string;
  access_key: string;
  allowed_model_ids?: number[];
  monthly_token_limit?: number;
  expires_at?: string;
  created_at: string;
  last_used_at?: string;
}

export interface ApiKeyCreate {
  name: string;
  description?: string;
  allowed_model_ids?: number[];
  monthly_token_limit?: number;
  expires_in_days?: number;
}

export interface ApiKeyCreateResponse extends ApiKey {
  api_key: string;
}
