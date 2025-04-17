# TODOs and Production Testing Plan (Ray Serve Architecture - Post-Refactor)

This document outlines remaining tasks and suggestions for testing the vLLM management backend and frontend, now architected for **Ray Serve**-based dynamic model serving.

---

## Outstanding TODOs / Potential Enhancements

*   **Ray Serve Dynamic Loading (On Inference):** Implement logic to load models *only when* an inference request targets them (currently loaded via serve toggle/redeploy).
*   **Ray Serve Idle Timeout:** Implement logic to automatically unload models after a configurable idle period.
*   **Management API/UI Enhancements:**
    *   Add UI/API to view detailed Ray Serve deployment status (replicas, etc.).
    *   Add UI/API for manual model unloading from Ray Serve.
    *   Add UI/API for removing models from the configuration file.
*   **Resource Management:** Add warnings or prevent serving models if estimated VRAM exceeds available resources.
*   **Monitoring:** Enhance monitoring UI for Ray Serve-specific metrics.
*   **Download Task Status:** Implement real-time download status updates in the UI.
*   **API Authentication:** Add authentication to management API if exposed.
*   **Process Management:** Consider using supervisor or systemd for backend/frontend/Ray head node for production robustness.
*   **Requirements Files:** Generate `requirements.txt` for backend and frontend. Update `start.sh` to use them.
*   **Testing:** See below for updated testing plan.

---

**Completed during Ray Serve Refactor:**
*   Replaced systemd/launcher with Ray Serve LLMApp.
*   `start.sh` installs Ray Serve and starts Ray cluster head.
*   Backend (`main.py`) initializes Ray and deploys LLMApp on startup.
*   Backend routers (`models_router.py`, `service_router.py`) refactored for Ray Serve (removed systemd, serve toggle triggers redeploy).
*   Frontend (`app.py`, `index.html`) updated (removed systemd controls, shows Ray Serve status).
*   Always-on base model logic added to `ray_deployments.py`.
*   Unified logging implemented for backend/Ray Serve (`/opt/vllm/logs/special_backend.log`).
*   Frontend logging standardized (`/opt/vllm/logs/frontend.log`).
*   Startup issues (Ray cluster conflict, ImportErrors) resolved.

---

## Production Testing Plan (Ray Serve)

1.  **Environment Setup:**
    *   Set up LXC, install NVIDIA drivers/CUDA, verify with `nvidia-smi`.
    *   Copy project files.
2.  **Run `start.sh`:**
    *   Execute `./start.sh`. Choose 'y' for initial cleanup if needed. Monitor for errors.
    *   Verify directories (`/opt/vllm`, `logs`, etc.) created.
    *   Ray Serve head node should be running (`ray status`), dashboard at `:8265`.
3.  **Configure & Download Models (via UI/API):**
    *   Access the Flask UI (`http://<lxc-ip>:5000`).
    *   Use the "Popular Models" list (or API) to add desired models (e.g., `nomic-embed-text-v1.5`, Llama 3) to the config using the "Add to Config" button. Verify they appear in the "Configured Models" table.
    *   Use the "Download" button for each newly added model. Monitor logs for download progress/completion. Verify status changes to "Downloaded".
4.  **Configure Models to Serve (via UI/API):**
    *   For the models you want Ray Serve to load (e.g., `nomic-embed-text-v1.5` and maybe one other), click the "Start Serving" button. Verify the status changes to "Serving".
    5.  **Ray Serve Redeploy (via Serve Toggle):**
        *   Toggling the "Serve" status for a model in the UI now automatically triggers a Ray Serve redeploy.
        *   Check the Ray Serve status via UI or `ray status`/dashboard. It should reflect the change in deployed models.
        *   Check the unified log file (`/opt/vllm/logs/special_backend.log`) for Ray Serve deployment logs.
6.  **Verify vLLM API:**
    *   Use `curl http://localhost:8000/v1/models` (or the UI's Test API button if implemented) to see which models Ray Serve *actually* loaded and is serving. Does this list match the models marked "serve: true" and downloaded?
7.  **OpenWebUI Integration:**
    *   Configure OpenWebUI's API endpoint URL to `http://<lxc-ip>:8000`.
    *   Verify OpenWebUI's model list populates correctly based on the previous step.
    *   Select different loaded models in OpenWebUI and run inference requests. Verify they work correctly.
8.  **Dynamic Loading/Unloading:**
    *   Test toggling "serve" status for models and redeploying. Verify models are loaded/unloaded in Ray Serve and reflected in `/v1/models`.
    *   (When implemented) Test idle timeout/unloading by leaving a model unused and checking if it is unloaded after the timeout.
9.  **Multi-GPU Testing:**
    *   If using multiple GPUs, check the TP size used by Ray Serve/vLLM. Is it appropriate for the largest model being served?
    *   Monitor GPU utilization (`nvidia-smi` or UI) during inference with different models.
    10. **Log Review:** Check the unified backend log (`/opt/vllm/logs/special_backend.log`) and frontend log (`/opt/vllm/logs/frontend.log`) for warnings/errors. Check Ray Serve dashboard and logs (`/tmp/ray/session_latest/logs/`) for deployment issues.
---

## Open Questions / TODOs

- **Ray Serve Dynamic Loading/Idle Timeout:** Implement these features.
- **Resource Management:** Implement VRAM checks before serving models.
- **Per-Model TP Size:** Note: Ray Serve's LLMApp might have limitations here; investigate alternatives if needed.
- **API Authentication:** Implement if needed.
- **Process Management:** Implement for production.
- **Monitoring:** Enhance UI.
- **Download Status:** Implement real-time updates.
- **Config Management:** Add model removal feature.

---

## References

- [Ray Serve LLM Docs](https://docs.ray.io/en/latest/serve/llm/serving-llms.html)
- [vLLM Docs](https://docs.vllm.ai/en/latest/)
- [Ray Serve API](https://docs.ray.io/en/latest/serve/api/index.html)