"""Deploy Apps API module.

This module provides endpoints for deploying and managing applications
like Open WebUI, n8n, Flowise, etc.

Module Structure:
- routes.py: Main API endpoint handlers (list, get, deploy)
- lifecycle.py: App lifecycle endpoints (start, stop, delete, logs)
- deployment.py: Background deployment logic
- utils.py: Helper functions and utilities
"""

from app.api.apps.routes import router

__all__ = ["router"]
