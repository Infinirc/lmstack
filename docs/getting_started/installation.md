# Installation

This guide covers the installation of all LMStack components.

## Backend Installation

### 1. Clone the Repository

```bash
git clone https://github.com/lmstack/lmstack.git
cd lmstack
```

### 2. Set Up Python Environment

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the backend directory:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/lmstack

# Security
SECRET_KEY=your-secret-key-here

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379
```

### 4. Initialize Database

```bash
# Run migrations
alembic upgrade head

# Create initial admin user
python -m app.scripts.create_admin
```

### 5. Start the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Frontend Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment

Create a `.env` file:

```bash
VITE_API_URL=http://localhost:8088
```

### 3. Start Development Server

```bash
npm run dev
```

### 4. Build for Production

```bash
npm run build
```

## Worker Installation

### 1. Install Docker

Follow the official Docker installation guide for your operating system.

### 2. Install NVIDIA Container Toolkit (for GPU support)

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Set Up Worker Agent

```bash
cd worker
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Worker

Create a `.env` file:

```bash
# Backend connection
BACKEND_URL=http://your-backend-server:8088
WORKER_TOKEN=your-worker-token

# Worker identification
WORKER_NAME=gpu-worker-01
```

### 5. Start Worker Agent

```bash
python agent.py
```

## Verifying Installation

### Check Backend Health

```bash
curl http://localhost:8088/health
```

Expected response:
```json
{"status": "healthy"}
```

### Check Worker Registration

After starting the worker agent, verify it appears in the backend:

```bash
curl http://localhost:8088/api/workers
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Deploy your first model
- [Configuration](configuration.md) - Advanced configuration options
