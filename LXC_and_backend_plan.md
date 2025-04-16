# vLLM + Ray Serve LXC Architecture & Setup (2025-04-16)

## Overview

This plan describes a modern LLM serving system using vLLM, Ray Serve, FastAPI, and Flask, running in an LXC container. The system is designed for dynamic, production-grade model serving with a management UI/API and OpenWebUI integration.

---

## Architecture

### Core Components

- **Ray Serve**: Orchestrates vLLM engine deployments. Handles dynamic model loading/unloading, request routing, and scaling.
- **vLLM**: High-performance LLM inference engine, managed as Ray Serve deployments.
- **FastAPI Backend**: Management API for model config, downloads, status, and Ray Serve control.
- **Flask Frontend**: Admin UI for managing models, downloads, and Ray Serve status.
- **OpenWebUI**: Connects to the Ray Serve OpenAI-compatible endpoint for model list and inference.

### Key Features

- **Always-On Model**: A base model (e.g., `nomic-ai/nomic-embed-text-v1.5`) is always loaded and available.
- **Load-on-Demand**: Other models are loaded into VRAM only when requested via API (e.g., by OpenWebUI).
- **Idle Timeout/Unloading**: On-demand models are automatically unloaded after a configurable period of inactivity (e.g., 10 minutes).
- **Unified OpenAI-Compatible Endpoint**: Ray Serve exposes a single endpoint (e.g., `:8000`) for OpenWebUI and other clients.
- **Management API/UI**: Used for model config, downloads, toggling "serve" status, and triggering Ray Serve redeploys.
- **Setup Script**: `start.sh` installs dependencies, starts Ray Serve, and launches the management backend/frontend.

---

## Setup Steps

1. **LXC Container Creation**
   - Launch Ubuntu/Debian LXC container.
   - Install NVIDIA drivers and CUDA toolkit (verify with `nvidia-smi`).

2. **Project Files**
   - Copy the entire project directory into the container.

3. **Run Setup Script**
   - `chmod +x start.sh`
   - `./start.sh`
   - Installs Python, Ray Serve, vLLM, FastAPI, Flask, and other dependencies.
   - Starts Ray Serve head node.
   - Starts backend (FastAPI) and frontend (Flask) management services.

4. **Model Management**
   - Use the admin UI to:
     - Browse/download popular models (filtered by VRAM).
     - Add models to config.
     - Toggle "serve" status for each model.
     - Trigger Ray Serve redeploy to apply changes.

5. **Ray Serve Deployment**
   - The backend (on startup or via redeploy endpoint) builds a Ray Serve LLMApp deployment:
     - Always includes the base model (e.g., `nomic-embed-text-v1.5`).
     - Includes any other models marked `"serve": true` and downloaded.
     - Ray Serve exposes an OpenAI-compatible API at `:8000`.

6. **OpenWebUI Integration**
   - Point OpenWebUI to the Ray Serve endpoint (`http://<lxc-ip>:8000`) for both model list and inference.
   - OpenWebUI will see the list of currently loaded models and can select any for inference.

7. **Dynamic Loading/Unloading**
   - When a new model is toggled to "serve: true" and redeploy is triggered, Ray Serve loads it.
   - Idle timeout logic (to unload models after inactivity) is managed by Ray Serve or a custom background task in the backend (to be implemented).

---

## Key Changes from Previous Plans

- **No systemd/launcher for vLLM**: Ray Serve now manages vLLM deployments.
- **No `active_model.txt`**: Model activation is handled by toggling "serve" in the config and redeploying.
- **No manual service activation**: All models marked "serve: true" are loaded at Ray Serve deploy/redeploy.
- **No direct systemd controls in UI**: All management is via Ray Serve and the management API.
- **Timeout/unloading**: To be implemented as a Ray Serve feature or custom backend logic.

---

## Open Questions / TODOs

- **Ray Serve Dynamic Loading**: Confirm best practices for dynamic model loading/unloading and idle timeout with Ray Serve and vLLM.
- **Resource Management**: How to handle VRAM exhaustion if too many models are marked "serve: true"?
- **Per-Model TP Size**: Ray Serve/vLLM currently uses a global TP size; per-model TP is not supported.
- **API Authentication**: Add authentication to management API if exposed.
- **Process Management**: Consider using supervisor or systemd for backend/frontend if needed.
- **Monitoring**: Enhance monitoring UI for Ray Serve deployments and resource usage.
- **Testing**: See `TODO_and_TESTING.md` for detailed testing plan.

---

## References

- [Ray Serve LLM Docs](https://docs.ray.io/en/latest/serve/llm/serving-llms.html)
- [vLLM Docs](https://docs.vllm.ai/en/latest/)
- [Ray Serve API](https://docs.ray.io/en/latest/serve/api/index.html)