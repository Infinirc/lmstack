# LMStack

[English](README.md)

LLM 部署管理平台 - 在分散式 GPU 節點上部署和管理大型語言模型。

## 功能特色

- 管理 Workers、模型和部署的 Web UI
- 支援 **vLLM** 和 **Ollama** 推論後端
- 基於 Docker 的 GPU Worker Agent
- 即時部署狀態監控
- OpenAI 相容 API 閘道

## 架構

```
┌─────────────────┐     ┌─────────────────┐
│   Web 前端      │────▶│   API 伺服器    │
│   (React)       │     │   (FastAPI)     │
└─────────────────┘     └────────┬────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
           ┌──────────────┐          ┌──────────────┐
           │ Worker Agent │          │ Worker Agent │
           │  (GPU 節點)  │          │  (GPU 節點)  │
           └──────────────┘          └──────────────┘
```

## 快速開始

### 前置需求

- Docker
- Docker Compose V2：`sudo apt install docker-compose-v2`
- Docker 權限設定：`sudo usermod -aG docker $USER && newgrp docker`
- 支援 CUDA 的 NVIDIA GPU
- NVIDIA Container Toolkit（使用 `./scripts/install-nvidia-toolkit.sh` 安裝）

### Docker Compose 部署

```bash
# 部署 Backend + Frontend
docker compose -f docker-compose.deploy.yml up -d
```

- 前端: http://localhost:3000
- 後端 API: http://localhost:8088

### 啟動 Worker（在 GPU 機器上）

```bash
docker run -d \
  --name lmstack-worker \
  --gpus all \
  --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e BACKEND_URL=http://你的伺服器IP:8088 \
  -e WORKER_NAME=gpu-worker-01 \
  infinirc/lmstack-worker:latest
```

### 使用方式

1. 使用 `admin` / `admin` 登入（首次登入後請更改密碼）
2. 查看 **Workers** 頁面 - Workers 會自動註冊
3. 在 **Models** 頁面新增模型
4. 在 **Deployments** 頁面建立部署
5. 使用 OpenAI 相容 API：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model": "llama3.2:3b", "messages": [{"role": "user", "content": "你好！"}]}'
```

## 開發環境

### 本地 Docker 構建

在本地構建並運行 Docker 映像：

```bash
# 構建所有映像
./scripts/build-local.sh

# 或構建特定映像
./scripts/build-local.sh backend
./scripts/build-local.sh frontend
./scripts/build-local.sh worker

# 運行本地構建的映像
docker compose -f docker-compose.local.yml up -d
```

### 不使用 Docker

```bash
# 終端機 1 - 前端
cd frontend
npm install
npm run dev

# 終端機 2 - 後端
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 終端機 3 - Worker（在 GPU 機器上）
cd worker
pip install -r requirements.txt
python agent.py --name gpu-worker-01 --server-url http://你的伺服器IP:8000
```

## API 文件

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 授權條款

Apache-2.0
