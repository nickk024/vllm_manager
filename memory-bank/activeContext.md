# Active Context

This file tracks the project's current status, including recent changes, current goals, and open questions.
2025-05-09 07:30:37 - Log of updates made.

*

## Current Focus
- Analyzing existing project structure (FastAPI backend, Flask frontend, Ray Serve for vLLM deployment) and documented plans (`PROJECT_ROADMAP.md`, `LXC_and_backend_plan.md`) to identify areas for further development.
- Specifically focusing on optimizing for a 3-GPU setup and enhancing Ollama-like ease of use.
- Preparing to use MCP to research best practices for vLLM multi-GPU deployment and Ollama-style CLI interactions.

*   

## Recent Changes
- [2025-05-09 07:31:08] - Memory Bank initialized.
- [2025-05-09 07:31:08] - Project goal and key features defined in `productContext.md`.
- [2025-05-09 07:35:34] - Completed comprehensive review of all project files.
- [2025-05-09 08:03:56] - Merged content from `LXC_and_backend_plan.md` into `productContext.md`.
- [2025-05-09 08:05:50] - Merging TODOs from `TODO_and_TESTING.md` into Active Context and System Patterns.

*

## Open Questions/Issues & TODOs (Consolidated)

**Ray Serve & vLLM Integration:**
*   **Dynamic Loading (On Inference):** Implement logic to load models *only when* an inference request targets them (currently loaded via serve toggle/redeploy). (from `TODO_and_TESTING.md`)
*   **Idle Timeout:** Implement logic to automatically unload models after a configurable idle period. (from `TODO_and_TESTING.md`)
*   Confirm best practices for dynamic model loading/unloading and idle timeout with Ray Serve and vLLM.
*   Per-model Tensor Parallelism (TP) size: Logic updated in `models_router.py` to better utilize 3 GPUs. Further testing and refinement might be needed for optimal performance across various model sizes.

**Resource Management:**
*   Add warnings or prevent serving models if estimated VRAM exceeds available resources. (from `TODO_and_TESTING.md`)
*   How to handle VRAM exhaustion if too many models are marked "serve: true" or requested simultaneously?
*   How will GPU resources (memory, compute) be allocated and managed dynamically for multiple models and concurrent requests across 3 GPUs?

**API & UI/UX Enhancements:**
*   **Management API/UI Enhancements (from `TODO_and_TESTING.md`):**
    *   Add UI/API to view detailed Ray Serve deployment status (replicas, etc.). (Partially addressed by adding backend endpoint and basic table in UI).
    *   Add UI/API for manual model unloading from Ray Serve. (Backend endpoint added, UI button added).
    *   Add UI/API for removing models from the configuration file. (Backend endpoint added, UI button added).
*   **Download Task Status:** Implement real-time download status updates in the UI. (Backend SSE implemented, frontend JS implemented).
*   What are the specific aspects of "Ollama-like ease of use" to replicate further in the UI? (User has clarified focus is on Web UI completion).
*   API Authentication: Add authentication to management API if exposed externally.

**Deployment & Operations:**
*   Process Management: Consider using supervisor or systemd for backend/frontend/Ray head node for production robustness.

**Monitoring:**
*   Enhance monitoring UI for Ray Serve-specific metrics. (Beyond basic GPU/System stats).

**Model Management (General):**
*   How will models be downloaded, stored, and managed (e.g., integration with Hugging Face Hub, local model cache)? (Current system uses HF Hub and local `MODELS_DIR`).

**Requirements & Build:**
*   **Requirements Files:** Generate `requirements.txt` for backend and frontend. Update `start.sh` to use them. (Noted in `TODO_and_TESTING.md`. `start.sh` currently installs from `backend/requirements.txt` and `frontend/requirements.txt` if they exist, otherwise installs a core list. Ensuring these files are accurate and complete is important).

**Testing:**
*   Refer to `memory-bank/systemPatterns.md` for the detailed production testing plan (migrated from `TODO_and_TESTING.md`).
*   Address items from `PROJECT_ROADMAP.md` under "Persistent Issues Needing Follow-up -> Test Compatibility Issues" and "Next Steps -> Immediate Priorities" related to testing (e.g., pytest-asyncio, better mocking).

*