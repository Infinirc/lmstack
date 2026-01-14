"""API Key utility functions and constants

Separated to avoid circular imports between api_keys routes and gateway service.
"""

import hashlib

API_KEY_PREFIX = "lmsk"


def hash_secret(secret: str) -> str:
    """Hash a secret key for storage"""
    return hashlib.sha256(secret.encode()).hexdigest()


def verify_secret(secret: str, hashed: str) -> bool:
    """Verify a secret against its hash"""
    return hash_secret(secret) == hashed
