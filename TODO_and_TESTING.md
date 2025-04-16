# TODOs and Production Testing Plan (Ray Serve Architecture)

This document outlines remaining tasks and suggestions for testing the vLLM management backend and frontend, now architected for **Ray Serve**-based dynamic model serving.

---

## Outstanding TODOs / Potential Enhancements

*   **Ray Serve Dynamic Loading/Unloading:** Implement logic (in Ray Serve deployment or backend) to load models on demand and unload after a configurable idle timeout (e.g., 10 minutes).
*   **Always-On Model:** Ensure the base model (e.g., `nomic-ai/nomic-embed-text-v1.5`) is always loaded at startup.
*   **Management API/UI:** Refactor endpoints and UI to:
    *   Toggle "serve" status for models.
    *   Trigger Ray Serve redeploys (to load/unload models).
    *   Remove systemd controls and activation logic.
    *   Display Ray Serve status and currently loaded models.
*   **OpenWebUI Integration:** Confirm OpenWebUI can connect to Ray Serve's OpenAI-compatible endpoint, see the list of loaded models, and send inference requests.
*   **Resource Management:** Add warnings or prevent toggling "serve" for too many models if VRAM is insufficient.
*   **Monitoring:** Enhance monitoring UI for Ray Serve deployments and resource usage.
*   **API Authentication:** Add authentication to management API if exposed.
*   **Process Management:** Consider using supervisor or systemd for backend/frontend if needed.
*   **Logging:** Ensure unified logging is robust and rotated.
*   **Requirements Files:** Generate `requirements.txt` for backend and frontend. Update `start.sh` to use them.
*   **Testing:** See below for updated testing plan.

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
5.  **Ray Serve Redeploy:**
    *   Click the "Restart Service" or "Redeploy" button in the UI to trigger a Ray Serve redeploy.
    *   Check the Ray Serve status via UI or `ray status`/dashboard. It should show the deployment as running.
    *   Check the unified log file (`logs/vllm_manager.log`) for Ray Serve deployment logs.
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
10. **Log Review:** Check the unified log file (`logs/vllm_manager.log`) for any warnings or errors during testing. Check Ray Serve dashboard and logs for deployment issues.
puppett
---

## Open Questions / TODOs

- **Ray Serve Dynamic Loading:** Confirm best practices for dynamic model loading/unloading and idle timeout with Ray Serve and vLLM.
- **Resource Management:** How to handle VRAM exhaustion if too many models are marked "serve: true"?
- **Per-Model TP Size:** Ray Serve/vLLM currently uses a global TP size; per-model TP is not supported.
- **API Authentication:** Add authentication to management API if exposed.
- **Process Management:** Consider using supervisor or systemd for backend/frontend if needed.
- **Monitoring:** Enhance monitoring UI for Ray Serve deployments and resource usage.
- **Testing:** Update this plan as new features are implemented.

---

## References

- [Ray Serve LLM Docs](https://docs.ray.io/en/latest/serve/llm/serving-llms.html)
- [vLLM Docs](https://docs.vllm.ai/en/latest/)
- [Ray Serve API](https://docs.ray.io/en/latest/serve/api/index.html)