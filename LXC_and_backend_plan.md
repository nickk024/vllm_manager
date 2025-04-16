# vLLM LXC + FastAPI Backend Architecture & Setup

## Architecture Overview

- **LXC Container**: Runs Ubuntu/Debian, hosts vLLM, model files, and the FastAPI backend.
- **vLLM**: Serves models via OpenAI-compatible API.
- **FastAPI Backend**: 
  - Maintains a list of available models (from config/model_config.json or a DB).
  - Provides REST API for OpenWebUI to fetch model list.
  - Triggers model downloads (calls download_models.py).
  - Switches/activates models (updates config, restarts vLLM).
- **Model Storage**: Models stored in /opt/vllm/models (or configurable path).
- **OpenWebUI**: Connects to FastAPI backend for model list, and to vLLM for inference.

## LXC Setup Instructions

1. **Create LXC Container**
   ```bash
   lxc launch images:ubuntu/22.04 vllm-lxc
   lxc exec vllm-lxc -- bash
   ```

2. **Install System Dependencies**
   ```bash
   apt update
   apt install -y python3-full python3-venv python3-dev build-essential git wget curl jq sudo
   apt install -y nvidia-driver-XXX nvidia-cuda-toolkit  # Replace XXX with your driver version
   ```

3. **Copy vLLM Installer & Scripts**
   - Copy the `vllm-installer` folder into the container (use `lxc file push` or `scp`).

4. **Run the Installer**
   ```bash
   cd vllm-installer
   bash vllm_installer.sh
   ```

5. **Install FastAPI Backend**
   ```bash
   python3 -m venv /opt/vllm/backend-venv
   source /opt/vllm/backend-venv/bin/activate
   pip install fastapi uvicorn pydantic python-multipart
   ```

6. **Expose Ports**
   - Ensure LXC container forwards ports for vLLM (default 8000) and FastAPI backend (e.g., 8080).

## Directory Structure (inside LXC)

```
/opt/vllm/
  ├── models/
  ├── config/
  │     └── model_config.json
  ├── scripts/
  │     ├── vllm_manage.sh
  │     ├── download_models.py
  │     └── ...
  ├── backend/
  │     ├── main.py
  │     └── ...
  ├── venv/
  └── logs/
```

## Next Steps

- Scaffold FastAPI backend in `/opt/vllm/backend/main.py`
- Backend will:
  - List models (GET /models)
  - Download models (POST /models/download)
  - Switch/activate model (POST /models/activate)
  - Serve model list to OpenWebUI (GET /models)
  - Optionally: Provide status, logs, etc.

---