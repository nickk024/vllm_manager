import os
import json
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, status as http_status
from typing import List, Optional, Dict, Any
from huggingface_hub import snapshot_download

# Import models, config using relative paths
from ..models import DownloadRequest, GeneralResponse
from ..config import load_model_config, CONFIG_PATH, MODELS_DIR

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Background Task Logic ---

def _download_single_model(model_name: str, model_config: Dict[str, Any], output_dir: str, token: Optional[str] = None, force: bool = False):
    """
    Internal function to download a single model.
    Logs success/failure. Returns True on success, False on failure.
    """
    model_dir = os.path.join(output_dir, model_name)
    logger.info(f"Preparing download for model: {model_name} to {model_dir}")

    # Check if model already exists and skip if not forcing
    if not force and os.path.exists(model_dir) and os.listdir(model_dir):
        logger.info(f"Model {model_name} already exists at {model_dir}. Skipping download.")
        # Ensure config.json exists within the model directory (might be missing if manually copied)
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
             logger.warning(f"config.json missing in existing model dir {model_dir}. Creating.")
             try:
                 # Ensure the config being written matches the one from model_config.json
                 # This assumes model_config passed here is the correct one from the main config
                 with open(config_path, "w") as f:
                     json.dump(model_config, f, indent=2)
             except Exception as e:
                 logger.error(f"Failed to create config.json for {model_name}: {e}")
                 # Decide if this should count as a failure for the download task
        return True # Count as success even if only config was created

    model_id = model_config.get("model_id")
    if not model_id:
        logger.error(f"Missing 'model_id' in config for model: {model_name}. Cannot download.")
        return False

    revision = model_config.get("revision", "main")

    # Make sure the target directory exists
    os.makedirs(model_dir, exist_ok=True)

    try:
        # Basic check if token is required (e.g., for meta models)
        # More robust checks might involve querying HF API if needed
        if "meta-llama" in model_id.lower() and not token:
            logger.error(f"Hugging Face token required for gated model {model_id}. Download skipped.")
            return False # Treat as failure

        logger.info(f"Starting download for {model_id} (revision: {revision}) to {model_dir}")
        # Perform the actual download
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=model_dir,
            token=token,
            # Consider adding ignore_patterns if needed, e.g., ignore_patterns=["*.safetensors"]
            # Consider using local_dir_use_symlinks=True/False based on filesystem/preference
        )

        # Create config.json within the model directory after successful download
        # This ensures the directory contains the necessary config for vLLM later
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        logger.info(f"Successfully downloaded and prepared {model_name}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {model_name} ({model_id}): {e}")
        # Optional: Clean up potentially incomplete download directory?
        # import shutil
        # shutil.rmtree(model_dir, ignore_errors=True)
        # logger.info(f"Cleaned up potentially incomplete download for {model_name} at {model_dir}")
        return False

def run_download_task(models_to_download: Dict[str, Any], output_dir: str, token: Optional[str], force: bool):
    """Background task function to download multiple models."""
    logger.info(f"Background download task started for {len(models_to_download)} models.")
    success_count = 0
    total_models = len(models_to_download)
    failed_models = []

    for model_name, config in models_to_download.items():
        logger.info(f"[{success_count+len(failed_models)+1}/{total_models}] Processing model in background: {model_name}")
        if _download_single_model(model_name, config, output_dir, token, force):
            success_count += 1
        else:
            failed_models.append(model_name)

    logger.info(f"Background download task finished.")
    logger.info(f"Successfully downloaded: {success_count}/{total_models}")
    if failed_models:
        logger.warning(f"Failed to download: {len(failed_models)}/{total_models} models: {', '.join(failed_models)}")
    # TODO: Implement notification mechanism here? (e.g., WebSocket, status endpoint)

# --- Route ---
@router.post("/models/download", response_model=GeneralResponse, summary="Download Models")
async def download_models_api(req: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Downloads specified models (or all configured models) in the background.
    Requires models to be present in model_config.json first. Use the
    /config/models endpoint to add models to the configuration.
    """
    model_configs = load_model_config()
    if not model_configs:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Model config file not found or empty at {CONFIG_PATH}. Add models using /config/models first."
        )

    models_to_process = {}
    if req.models:
        # Download only specified models (must exist in config)
        not_found = []
        for name in req.models:
            if name in model_configs:
                models_to_process[name] = model_configs[name]
            else:
                logger.warning(f"Requested model '{name}' for download not found in config {CONFIG_PATH}. Skipping.")
                not_found.append(name)
        if not models_to_process:
             error_detail = "None of the requested models for download found in configuration."
             if not_found:
                  error_detail += f" Models not found: {', '.join(not_found)}"
             raise HTTPException(
                 status_code=http_status.HTTP_400_BAD_REQUEST,
                 detail=error_detail
            )
    else:
        # Download all models currently in the config file
        models_to_process = model_configs
        if not models_to_process:
             raise HTTPException(
                 status_code=http_status.HTTP_400_BAD_REQUEST,
                 detail="No models configured for download in {CONFIG_PATH}."
            )

    # Add the download task to run in the background
    background_tasks.add_task(run_download_task, models_to_process, MODELS_DIR, req.token, req.force)

    return GeneralResponse(
        status="ok",
        message=f"Download task started in background for {len(models_to_process)} models."
    )