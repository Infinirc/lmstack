<p align="center">
  <img src="docs/LMStack-light.png" alt="LMStack" height="80">
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

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
           │ Worker       │          │ Worker       │
           │  (GPU 節點)  │           │  (GPU 節點)  │
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
- 後端 API: http://localhost:52000

### Windows Docker Desktop - 區域網路存取

Windows 防火牆預設會阻擋區域網路存取。請選擇以下其中一種方式：

**方式一：關閉防火牆（最簡單）**

```powershell
# 在 PowerShell（系統管理員）中執行
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
```

**方式二：新增防火牆規則（較安全）**

```powershell
# 在 PowerShell（系統管理員）中執行
# 基本端口（前端 + 後端 API）
New-NetFirewallRule -DisplayName "LMStack" -Direction Inbound -LocalPort 3000,52000 -Protocol TCP -Action Allow

# 模型部署端口（依需求新增，例如 40000-40100）
New-NetFirewallRule -DisplayName "LMStack Models" -Direction Inbound -LocalPort 40000-40100 -Protocol TCP -Action Allow

# App 端口（例如 Open WebUI 使用 46488）
New-NetFirewallRule -DisplayName "LMStack Apps" -Direction Inbound -LocalPort 46000-46500 -Protocol TCP -Action Allow
```

> **注意**：部署模型或 App 時，請在 UI 中查看分配的端口，並確保該端口已在防火牆中開放。

### 使用方式

1. 使用 `admin` / `admin` 登入（首次登入後請更改密碼）
2. 前往 **Workers** 頁面，點擊 **Add Worker** 取得 Docker 指令
3. 在 GPU 機器上執行該 Docker 指令以註冊 Worker
4. 在 **Models** 頁面新增模型
5. 在 **Deployments** 頁面建立部署
6. 使用 OpenAI 相容 API：

```bash
curl http://localhost:52000/v1/chat/completions \
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

# 運行本地構建的 backend + frontend
docker compose -f docker-compose.local.yml up -d
```

然後前往 UI 中的 **Workers** 頁面新增 Worker。

### 不使用 Docker

```bash
# 終端機 1 - 前端
cd frontend
npm install
npm run dev

# 終端機 2 - 後端
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 52000 --reload

# 終端機 3 - Worker（在 GPU 機器上）
cd worker
pip install -r requirements.txt
python agent.py --name gpu-worker-01 --server-url http://你的伺服器IP:52000
```

## API 文件

- Swagger UI: http://localhost:52000/docs
- ReDoc: http://localhost:52000/redoc

## 授權條款

Apache-2.0
