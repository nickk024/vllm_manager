# System Patterns

This file documents recurring patterns, standards, and key procedures used in the project.
*Last Updated: 2025-05-09 08:06:51 (Incorporating TODO_and_TESTING.md)*

*

## Coding Patterns

*   (To be defined)

## Architectural Patterns

*   **Service Orchestration:** Ray Serve is used to manage and deploy vLLM instances as scalable services.
*   **API Layer:** FastAPI provides the backend management API, while Flask serves the frontend UI.
*   **Configuration Management:** Model configurations and serving states are managed via `model_config.json`.
*   **Background Tasks:** FastAPI's `BackgroundTasks` are used for operations like model downloads to keep the API responsive.
*   **Real-time Updates:** Server-Sent Events (SSE) are used to provide real-time updates from the backend to the frontend (e.g., download status).

## Testing Patterns

### Production Testing Plan (Ray Serve Architecture)

This plan outlines steps for testing the vLLM management backend and frontend, specifically with the Ray Serve-based dynamic model serving architecture.

1.  **Environment Setup:**
    *   Set up LXC container (or target production-like environment).
    *   Install NVIDIA drivers and CUDA toolkit. Verify with `nvidia-smi`.
    *   Copy the entire project directory into the container/environment.

2.  **Run `start.sh` Script:**
    *   Make executable: `chmod +x start.sh`.
    *   Execute: `./start.sh`.
    *   If prompted for cleanup of a previous installation, choose 'y' for a clean test.
    *   Monitor script output for any errors during dependency installation or service startup.
    *   Verify directories (`/opt/vllm` or custom `VLLM_HOME`, `logs`, `config`, `models`) are created correctly.
    *   Verify Ray head node is running (e.g., using `ray status` in the venv). The Ray Dashboard should be accessible (typically `http://localhost:8265` if port forwarded).
    *   Verify backend FastAPI service starts (check logs, `ps aux | grep uvicorn`).
    *   Verify frontend Flask service starts (check logs, `ps aux | grep flask`).

3.  **Configure & Download Models (via UI/API):**
    *   Access the Flask Admin UI (e.g., `http://<lxc-ip>:5000`).
    *   Navigate to the "Popular Models" section.
    *   Select a few models (e.g., a small one like `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and a larger one like `meta-llama/Llama-3-8B-Instruct` if VRAM allows).
    *   Click "Add to Config" for each. Verify they appear in the "Configured Models" table with "Downloaded: No" and "Serve Status: Not Serving".
    *   For each newly added model, click its "Download" button.
        *   If it's a gated model, the download form should appear; test with and without a valid token.
        *   Monitor the UI for download status updates (queued, downloading, completed/failed/skipped).
        *   Check backend logs (`vllm_manager_app.log`) for download progress and any errors.
        *   Verify the model files appear in the `models/<model_config_key>/` directory.
        *   Verify the UI updates to "Downloaded: Yes" upon successful completion.

4.  **Configure Models to Serve (via UI/API):**
    *   For a downloaded model, click the "Start Serving (Redeploy)" button.
    *   Verify the "Serve Status" in the UI changes to "Serving".
    *   Check the Ray Serve status via the UI's "Ray Serve Status" section (including individual deployments) or by using `ray status` and checking the Ray Dashboard. The new model should appear as a running deployment.
    *   Check backend logs for Ray Serve deployment logs related to this model.

5.  **Verify vLLM API (OpenAI-Compatible Endpoint):**
    *   Use the "Test API" button in the UI (if available) or `curl http://localhost:8000/v1/models` (or the appropriate IP/port for your LXC).
    *   The response should list the models currently being served by Ray Serve. This list should match the models marked "Serve Status: Serving" in the UI.
    *   Test an inference request against a served model using a tool like `curl` or Postman to the `http://localhost:8000/v1/chat/completions` endpoint (or the specific route for your model if not using a unified router).

6.  **OpenWebUI Integration (If Applicable):**
    *   Configure OpenWebUI's API endpoint URL to `http://<lxc-ip>:8000`.
    *   Verify OpenWebUI's model list populates correctly with the actively served models.
    *   Select different loaded models in OpenWebUI and run inference requests. Verify correct responses.

7.  **Dynamic Loading/Unloading & Redeployment:**
    *   **Stop Serving:** For a model that is "Serving", click "Stop Serving (Redeploy)". Verify its status changes to "Not Serving" in the UI. Check Ray Serve (dashboard/API) to confirm the deployment is removed or scaled down (depending on implementation). The `/v1/models` endpoint should no longer list it.
    *   **Force Unload:** For a model that is "Serving", click "Force Unload". Verify its status changes to "Not Serving" and it's removed from Ray Serve deployments. The config file should also reflect `serve: false`.
    *   **Add and Serve New Model:** Add a new model to config, download it, then click "Start Serving". Verify it gets deployed to Ray Serve and becomes available.
    *   **(Future) Idle Timeout:** Once implemented, test by leaving a model unused and checking if it's automatically unloaded after the configured timeout.

8.  **Multi-GPU Testing (Specific to 3-GPU Setup):**
    *   Configure a large model (e.g., Llama-3-70B if VRAM allows, or a 30B+ model) and ensure `tensor_parallel_size` is set to 3 in `model_config.json` (either manually or via the updated "Add to Config" logic).
    *   Serve this model.
    *   Monitor GPU utilization across all 3 GPUs using `nvidia-smi` or the UI's monitoring section during inference. Check for balanced load if possible, or at least that all 3 GPUs are being utilized for that model.
    *   Test serving multiple smaller models simultaneously, each potentially on a different GPU or sharing GPUs if TP=1 for them. Observe resource allocation.

9.  **Resource Management Testing:**
    *   Attempt to configure and serve more models than the available VRAM can handle. Observe system behavior and any warnings/errors from the backend or Ray Serve. (Relates to "Resource Management" TODO).

10. **Log Review:**
    *   Thoroughly review backend logs (`/opt/vllm/logs/vllm_manager_app.log` or as configured).
    *   Check Ray Serve logs (typically in `/tmp/ray/session_latest/logs/`).
    *   Look for any errors, warnings, or unexpected behavior during all test phases.

11. **Stress Testing (from `PROJECT_ROADMAP.md`):**
    *   Simulate high-load scenarios with concurrent requests to served models.
    *   Perform long-running stability tests to check for memory leaks or performance degradation over time.

---
*Log of updates:*
*2025-05-09 07:30:51 - Initial file creation.*
*2025-05-09 08:06:19 - Added Production Testing Plan from `TODO_and_TESTING.md` and initial Architectural Patterns based on code review.*
*2025-05-09 08:06:51 - Content from TODO_and_TESTING.md (Production Testing Plan) merged.*