# Getting Started

Welcome to LMStack! This section will help you get up and running with LMStack quickly.

## Overview

LMStack consists of three main components:

1. **Backend API Server**: A FastAPI-based server that manages deployments, workers, and provides the API endpoints
2. **Frontend Web UI**: A React-based dashboard for managing and monitoring your LLM deployments
3. **Worker Agents**: Lightweight agents that run on GPU nodes and manage Docker containers for model inference

## Prerequisites

Before installing LMStack, ensure you have:

- **Python 3.10+** for the backend and worker agents
- **Node.js 18+** for the frontend
- **Docker** installed on all worker nodes
- **NVIDIA Container Toolkit** (for GPU support)
- **PostgreSQL 13+** for the database

## System Requirements

### Backend Server
- CPU: 2+ cores
- RAM: 4GB minimum
- Storage: 10GB for application

### Worker Nodes
- GPU: NVIDIA GPU with CUDA support (recommended)
- RAM: 16GB+ (depends on model size)
- Storage: 100GB+ for model weights

## Next Steps

1. [Installation](installation.md) - Install all components
2. [Quick Start](quickstart.md) - Deploy your first model
3. [Configuration](configuration.md) - Customize your setup
