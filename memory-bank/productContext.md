# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.

*
## Project Overview (from PROJECT_ROADMAP.md)
vLLM Manager is a comprehensive management system for deploying, serving, and monitoring large language models using vLLM and Ray Serve. The project consists of a backend API service and a frontend web interface.

## Project Goal
- To create a server application that leverages vLLM to manage and run a collection of large language models, supporting both simultaneous and sequential execution across a multi-GPU (specifically 3 GPUs, but adaptable) setup.
- The server should aim for the ease of use characteristic of Ollama while retaining the power and performance of vLLM.
- The system is designed for dynamic, production-grade model serving with a management UI/API and potential OpenWebUI integration.

*

## Key Features (Consolidated)
- **Dynamic Model Management**:
    - Load/unload models on demand.
    - Toggle "serve" status for models via API/UI.
    - Always-on base model capability (e.g., for embeddings).
    - Idle timeout and automatic unloading of inactive models (planned).
- **Multi-GPU Support**:
    - Specifically target 3-GPU setups, allowing models to utilize tensor parallelism across these GPUs.
    - Efficient resource allocation and management for multiple models and concurrent requests.
- **Serving & API**:
    - High-performance LLM inference using vLLM.
    - Unified OpenAI-compatible API endpoint exposed by Ray Serve (e.g., `:8000`) for clients like OpenWebUI.
    - Management API (FastAPI) for model configuration, downloads, status, and Ray Serve control.
- **User Experience**:
    - Admin UI (Flask) for managing models, downloads, and Ray Serve status.
    - Aim for Ollama-like ease of use (e.g., simple model pulling/running paradigm, though primarily via UI/API in current implementation).
- **Deployment & Setup**:
    - Designed to run in an LXC container.
    - `start.sh` script for dependency installation, Ray Serve startup, and launching management backend/frontend.

*

## Overall Architecture (from LXC_and_backend_plan.md)

### Core Components
- **Ray Serve**: Orchestrates vLLM engine deployments. Handles dynamic model loading/unloading, request routing, and scaling.
- **vLLM**: High-performance LLM inference engine, managed as Ray Serve deployments.
- **FastAPI Backend**: Management API for model config, downloads, status, and Ray Serve control.
- **Flask Frontend**: Admin UI for managing models, downloads, and Ray Serve status.
- **OpenWebUI**: Connects to the Ray Serve OpenAI-compatible endpoint for model list and inference (intended integration).

### Key Architectural Decisions/Changes (from LXC_and_backend_plan.md)
- **Ray Serve as Primary Orchestrator**: Ray Serve manages vLLM deployments, replacing previous systemd/launcher approaches.
- **Configuration-Driven Serving**: Model activation is handled by toggling a "serve" flag in `model_config.json` and triggering a Ray Serve redeploy. All models marked "serve: true" are loaded/reloaded by Ray Serve.
- **Centralized Management**: All management actions are intended to be via the FastAPI/Flask management layer interacting with Ray Serve.

*

---
*Log of updates:*
*2025-05-09 07:30:26 - Initial Memory Bank creation and population of project goal/features.*
*2025-05-09 08:02:53 - Integrated content from `LXC_and_backend_plan.md` into Project Goal, Key Features, and Overall Architecture. Added details on Ray Serve orchestration, dynamic loading, and management UI/API focus.*
*2025-05-09 08:04:54 - Added Project Overview from `PROJECT_ROADMAP.md`.*