"""App database model for deployable applications like Open WebUI"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.api_key import ApiKey
    from app.models.worker import Worker


class AppType(str, Enum):
    """Available app types"""

    OPEN_WEBUI = "open-webui"
    N8N = "n8n"
    FLOWISE = "flowise"
    ANYTHINGLLM = "anythingllm"
    LOBECHAT = "lobechat"


class AppStatus(str, Enum):
    """App deployment status"""

    PENDING = "pending"
    PULLING = "pulling"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# App definitions with default configurations
APP_DEFINITIONS = {
    AppType.OPEN_WEBUI: {
        "name": "Open WebUI",
        "description": "A user-friendly WebUI for LLMs, originally designed for Ollama",
        "image": "ghcr.io/open-webui/open-webui:main",
        "internal_port": 8080,
        "env_template": {
            "OPENAI_API_BASE_URL": "{lmstack_api_url}",
            "OPENAI_API_KEY": "{api_key}",
            "OLLAMA_BASE_URL": "",  # Disable Ollama
            "WEBUI_SECRET_KEY": "",
            "ENABLE_OLLAMA_API": "false",
        },
        "volumes": [{"name": "open-webui-data", "destination": "/app/backend/data"}],
    },
    AppType.N8N: {
        "name": "n8n",
        "description": "Workflow automation tool with 400+ integrations and AI capabilities",
        "image": "docker.n8n.io/n8nio/n8n",
        "internal_port": 5678,
        "env_template": {
            "N8N_SECURE_COOKIE": "false",  # Allow HTTP access (required for non-HTTPS setups)
            "N8N_EDITOR_BASE_URL": "{app_url}",  # Required for WebSocket origin check
            "N8N_HOST": "{app_host_port}",  # Host with port for WebSocket origin validation
            "WEBHOOK_URL": "{app_url}",  # Webhook URL for external triggers
            "N8N_PROXY_HOPS": "1",  # Trust proxy headers (nginx)
            "GENERIC_TIMEZONE": "UTC",
        },
        "volumes": [{"name": "n8n-data", "destination": "/home/node/.n8n"}],
    },
    AppType.FLOWISE: {
        "name": "Flowise",
        "description": "Build AI Agents and LLM workflows visually with drag & drop",
        "image": "flowiseai/flowise",
        "internal_port": 3000,
        "env_template": {
            "FLOWISE_SECRETKEY_OVERWRITE": "{secret_key}",  # Consistent encryption key for credentials
        },
        "volumes": [{"name": "flowise-data", "destination": "/root/.flowise"}],
    },
    AppType.ANYTHINGLLM: {
        "name": "AnythingLLM",
        "description": "All-in-one AI app with RAG, agents, and multi-user support",
        "image": "mintplexlabs/anythingllm",
        "internal_port": 3001,
        "env_template": {
            "STORAGE_DIR": "/app/server/storage",
            "LLM_PROVIDER": "generic-openai",
            "GENERIC_OPEN_AI_BASE_PATH": "{lmstack_api_url}",
            "GENERIC_OPEN_AI_API_KEY": "{api_key}",
            "GENERIC_OPEN_AI_MODEL_PREF": "default",
            "GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT": "8192",
        },
        "volumes": [{"name": "anythingllm-data", "destination": "/app/server/storage"}],
        "cap_add": ["SYS_ADMIN"],  # Required for AnythingLLM
    },
    AppType.LOBECHAT: {
        "name": "LobeChat",
        "description": "Modern AI chat framework with multi-provider support and plugins",
        "image": "lobehub/lobe-chat",
        "internal_port": 3210,
        "env_template": {
            "ENABLED_OPENAI": "0",  # Disable OpenAI
            "ENABLED_VLLM": "1",  # Enable vLLM provider
            "VLLM_API_KEY": "{api_key}",
            "VLLM_PROXY_URL": "{lmstack_api_url}",  # With /v1
            "VLLM_MODEL_LIST": "{model_list}",  # Auto-populated with running models
            "ACCESS_CODE": "{secret_key}",  # Access password for security
        },
        "volumes": [],  # LobeChat is stateless by default
    },
}


class App(Base):
    """Deployed application instance"""

    __tablename__ = "apps"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # App type (e.g., open-webui)
    app_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Display name
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Worker where app is deployed
    worker_id: Mapped[int] = mapped_column(Integer, ForeignKey("workers.id"), nullable=False)

    # Associated API key (auto-created for the app)
    api_key_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("api_keys.id"), nullable=True
    )

    # Status
    status: Mapped[str] = mapped_column(String(50), default=AppStatus.PENDING.value)
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Container info
    container_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Proxy path (e.g., /apps/open-webui)
    proxy_path: Mapped[str] = mapped_column(String(255), nullable=False)

    # Whether to use nginx proxy on controller (True) or direct worker connection (False)
    use_proxy: Mapped[bool] = mapped_column(default=True)

    # Custom configuration (overrides defaults)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    worker: Mapped["Worker"] = relationship("Worker")
    api_key: Mapped[Optional["ApiKey"]] = relationship("ApiKey")

    def __repr__(self) -> str:
        return f"<App(id={self.id}, type='{self.app_type}', status='{self.status}')>"
