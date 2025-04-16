# TODOs and Production Testing Plan

This document outlines remaining tasks and suggestions for testing the vLLM management backend and frontend in a production-like environment (e.g., the target LXC container).

## Outstanding TODOs / Potential Enhancements

*   **API Test Endpoint:** The `/service/test-vllm-api` endpoint was added to the service router (`service_router.py`) but needs a corresponding button/display in the Flask frontend (`frontend/templates/index.html` and `frontend/app.py`) if desired.
*   **Error Handling:** Review error handling in both backend and frontend for edge cases (e.g., backend down, specific `systemctl` failures, download errors, config file corruption). The `call_backend` helper in Flask could be made more robust.
*   **Security - `sudo` Permissions:** The current method of using passwordless `sudo` for `systemctl` in the backend API is convenient but carries security risks. **Strongly consider** alternatives for production:
    *   A dedicated, minimal privileged helper script/daemon.
    *   Using a message queue where the backend sends requests and a separate privileged process executes them.
    *   Running the backend API itself as a dedicated system user with *only* the necessary `systemctl` permissions granted via `sudoers`.
*   **Security - API Authentication:** The backend API currently has no authentication. Add API key authentication or another mechanism if exposing it beyond localhost.
*   **Configuration Management:**
    *   Allow editing/removing models from `model_config.json` via the API/UI.
    *   Make backend/frontend ports and the backend URL configurable via environment variables more consistently (partially done).
    *   Consider storing configuration (like `active_model.txt`) in a more robust way if needed (e.g., small database, dedicated config service).
*   **Download Task Status:** The background download task currently only logs completion/errors. Implement a way for the frontend to poll for or receive real-time status updates (e.g., via WebSockets or a status endpoint).
*   **Frontend Polish:** The Flask UI is basic. Improve styling, add loading indicators for actions, provide more detailed feedback.
*   **Process Management:** The `start.sh` script uses `nohup` to run backend/frontend. For production, use a proper process manager like `supervisor` or run them as systemd services (potentially user services) for better reliability and management.
*   **Logging:** Review log levels and messages. Consider structured logging (JSON) for easier parsing. Ensure log rotation settings are appropriate.
*   **Resource Usage:** The `get_system_stats` uses `top` and `free`, which might not be the most efficient or cross-platform way. Consider using the `psutil` Python library if adding dependencies is acceptable.
*   **Dependencies:** Create `requirements.txt` files for both backend and frontend based on the final imports. Update `start.sh` to use them.

## Production Testing Plan

1.  **Environment Setup:**
    *   Set up the target LXC container with Ubuntu/Debian.
    *   Install correct NVIDIA drivers and CUDA toolkit **manually**. Verify with `nvidia-smi`.
    *   Copy the project files into the LXC.
2.  **Run `start.sh`:**
    *   Execute `./start.sh`. Monitor output for errors during dependency installation, venv setup, and service configuration.
    *   Verify directories (`/opt/vllm`, logs, config, models, venv) are created with correct ownership (`whoami` user).
3.  **Configure `sudoers`:**
    *   **Carefully** edit `/etc/sudoers` (using `sudo visudo`) to grant the user running the backend passwordless permission *only* for the necessary `systemctl` commands related to the `vllm` service, as shown in the `start.sh` output. Test this permission manually (`sudo -l -U <username>`).
4.  **Verify Services:**
    *   Check if the backend (uvicorn) and frontend (flask) processes are running (`ps aux | grep uvicorn`, `ps aux | grep flask`).
    *   Check the logs in `/opt/vllm/logs/` for startup errors.
    *   Check systemd status: `sudo systemctl status vllm`. It might be inactive initially.
5.  **Frontend UI Testing:**
    *   Access the Flask UI (`http://<lxc-ip>:5000`).
    *   Verify initial status display (service status, models list, monitoring stats). Are GPU stats showing correctly?
    *   **Model Configuration:** Use the API (e.g., via `curl` or modifying the frontend) to add a popular model using `/api/v1/config/models`. Verify `model_config.json` is updated.
    *   **Model Download:** Trigger a download for a configured model via the UI. Monitor backend logs (`/opt/vllm/logs/vllm_backend.log`) for progress/errors. Verify model files appear in `/opt/vllm/models/<model_key>/`. Test downloading gated models (requires passing HF token if UI is updated, or setting `HF_TOKEN` env var for backend).
    *   **Service Control:** Use UI buttons to:
        *   Start the service. Verify `sudo systemctl status vllm` shows active. Check backend logs.
        *   Stop the service. Verify status changes.
        *   Enable the service. Verify with `sudo systemctl is-enabled vllm`.
        *   Disable the service. Verify status.
    *   **Model Activation:**
        *   Ensure at least two different models (ideally with different TP requirements) are configured and downloaded.
        *   Activate Model A using the UI button. Verify `active_model.txt` contains Model A's key. Verify `sudo systemctl status vllm` shows active (or restarted). Check launcher logs in systemd journal (`sudo journalctl -u vllm -f`) to see if it logged loading Model A with the correct TP size. Test inference via vLLM API (port 8000).
        *   Activate Model B. Verify `active_model.txt` updates. Verify service restarts. Check launcher logs for Model B and its TP size. Test inference.
    *   **Monitoring:** Refresh the UI and check if monitoring stats update and look reasonable.
6.  **Multi-GPU Testing:**
    *   If using multiple GPUs, specifically check the `tensor_parallel_size` logged by `vllm_launcher.py` (via `sudo journalctl -u vllm`) when activating models of different sizes. Does it match the value calculated and stored in `model_config.json`?
    *   Check GPU utilization across multiple GPUs during inference using `nvidia-smi` or the monitoring endpoint.
7.  **Stress Testing (Optional):**
    *   Send multiple concurrent requests to the vLLM API.
    *   Trigger downloads while inference is running.
    *   Monitor resource usage (CPU, RAM, GPU VRAM) under load.
8.  **Log Review:** Check both backend and frontend logs for any warnings or errors during testing. Check systemd journal for vLLM service logs.