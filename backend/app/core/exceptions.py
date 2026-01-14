"""Custom exceptions for LMStack API.

This module provides standardized exceptions for consistent error handling
across the application.
"""

from typing import Any, Optional


class LMStackError(Exception):
    """Base exception for all LMStack errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "internal_error",
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to API response format."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                **self.details,
            }
        }


class NotFoundError(LMStackError):
    """Resource not found."""

    def __init__(self, resource: str, identifier: Any = None):
        message = f"{resource} not found"
        if identifier is not None:
            message = f"{resource} '{identifier}' not found"
        super().__init__(
            message=message,
            status_code=404,
            error_type="not_found",
            details={"resource": resource},
        )


class ValidationError(LMStackError):
    """Request validation failed."""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            status_code=400,
            error_type="validation_error",
            details=details,
        )


class AuthenticationError(LMStackError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=401,
            error_type="authentication_error",
        )


class AuthorizationError(LMStackError):
    """Authorization failed (insufficient permissions)."""

    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            message=message,
            status_code=403,
            error_type="authorization_error",
        )


class ConflictError(LMStackError):
    """Resource conflict (e.g., duplicate name)."""

    def __init__(self, message: str, resource: Optional[str] = None):
        details = {"resource": resource} if resource else {}
        super().__init__(
            message=message,
            status_code=409,
            error_type="conflict_error",
            details=details,
        )


class WorkerError(LMStackError):
    """Worker communication or operation error."""

    def __init__(self, message: str, worker_id: Optional[int] = None):
        details = {"worker_id": worker_id} if worker_id else {}
        super().__init__(
            message=message,
            status_code=502,
            error_type="worker_error",
            details=details,
        )


class DeploymentError(LMStackError):
    """Deployment operation error."""

    def __init__(self, message: str, deployment_id: Optional[int] = None):
        details = {"deployment_id": deployment_id} if deployment_id else {}
        super().__init__(
            message=message,
            status_code=500,
            error_type="deployment_error",
            details=details,
        )


class DockerError(LMStackError):
    """Docker operation error."""

    def __init__(self, message: str, container_id: Optional[str] = None):
        details = {"container_id": container_id} if container_id else {}
        super().__init__(
            message=message,
            status_code=500,
            error_type="docker_error",
            details=details,
        )


class ExternalServiceError(LMStackError):
    """External service (HuggingFace, Ollama, etc.) error."""

    def __init__(self, service: str, message: str):
        super().__init__(
            message=f"{service} error: {message}",
            status_code=502,
            error_type="external_service_error",
            details={"service": service},
        )
