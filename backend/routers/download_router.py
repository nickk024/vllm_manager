import os
import json # For SSE data
import logging
import asyncio # For SSE event
import functools # For partial in run_in_executor

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, status as http_status # Added Request
from starlette.responses import EventSourceResponse # For SSE
from typing import List, Optional, Dict, Any
from huggingface_hub import snapshot_download

# Import models, config using relative paths
from ..models import DownloadRequest, GeneralResponse
from ..config import load_model_config, CONFIG_PATH, MODELS_DIR

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Download Status Tracking (In-memory) ---
# Structure: { "model_config_key": {"status": "queued/downloading/completed/failed/skipped", "message": "details", "timestamp": "..."} }
download_statuses: Dict[str, Dict[str, Any]] = {} # Value can store more than just string
status_event = asyncio.Event() # To signal status changes for SSE

async def update_download_status(model_name_key: str, status: str, message: Optional[str] = None, progress: Optional[float] = None):
    """Helper to update status and set event."""
    global download_statuses
    if model_name_key not in download_statuses:
        download_statuses[model_name_key] = {}
    
    download_statuses[model_name_key]["status"] = status
    if message:
        download_statuses[model_name_key]["message"] = message
    else:
        download_statuses[model_name_key].pop("message", None)
    
    if progress is not None:
        download_statuses[model_name_key]["progress"] = progress
    else:
        # Remove progress if status is not 'downloading' or if explicitly set to None
        if status != "downloading":
            download_statuses[model_name_key].pop("progress", None)

    download_statuses[model_name_key]["timestamp"] = asyncio.get_event_loop().time() # Store as float for easier comparison

    logger.debug(f"Download status for {model_name_key}: {download_statuses[model_name_key]}")
    status_event.set()
    await asyncio.sleep(0) # Yield control to allow event to be processed
    status_event.clear()

# --- Background Task Logic ---

async def _download_single_model(model_name: str, model_config: Dict[str, Any], output_dir: str, token: Optional[str] = None, force: bool = False):
    """
    Internal async function to download a single model.
    Logs success/failure. Returns True on success, False on failure.
    Updates global download_statuses.
    """
    await update_download_status(model_name, "queued", f"Download for {model_name} is queued.")
    model_dir = os.path.join(output_dir, model_name)
    logger.info(f"Preparing download for model: {model_name} to {model_dir}")

    # Check if model already exists and skip if not forcing
    if not force and os.path.exists(model_dir) and os.listdir(model_dir):
        logger.info(f"Model {model_name} already exists at {model_dir}. Skipping download.")
        await update_download_status(model_name, "skipped", f"Model already exists at {model_dir}.")
        return True # Existing model, count as success

    model_id = model_config.get("model_id")
    if not model_id:
        logger.error(f"Missing 'model_id' in config for model: {model_name}. Cannot download.")
        await update_download_status(model_name, "failed", "Missing 'model_id' in config.")
        return False

    revision = model_config.get("revision", "main")

    # Make sure the target directory exists
    os.makedirs(model_dir, exist_ok=True)

    try:
        # Basic check if token is required (e.g., for meta models)
        # More robust checks might involve querying HF API if needed
        if "meta-llama" in model_id.lower() and not token:
            logger.error(f"Hugging Face token required for gated model {model_id}. Download skipped.")
            await update_download_status(model_name, "failed", f"Token required for gated model {model_id}.")
            return False # Treat as failure

        logger.info(f"Starting download for {model_id} (revision: {revision}) to {model_dir}")
        await update_download_status(model_name, "downloading", f"Downloading {model_id} (revision: {revision})...", progress=0) # Initial progress

        # snapshot_download is synchronous. We run it in a thread pool.
        # For actual progress reporting, snapshot_download would need to support a callback,
        # or we'd need to use a different download mechanism (like hf_transfer with progress hooks).
        # This example simulates progress updates before/after the blocking call.
        loop = asyncio.get_event_loop()
        try:
            # Using functools.partial to correctly pass keyword arguments
            await loop.run_in_executor(
                None,
                functools.partial(
                    snapshot_download,
                    repo_id=model_id,
                    revision=revision,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False, # Explicitly set for clarity
                    token=token,
                    force_download=force # Pass the force parameter
                    # Add other relevant snapshot_download params if needed from model_config
                    # e.g., allow_patterns=model_config.get("allow_patterns"),
                    # ignore_patterns=model_config.get("ignore_patterns")
                )
            )
            logger.info(f"Successfully downloaded {model_name}")
            # Status will be updated to "completed" by the calling task
            return True
        except Exception as e: # Catch specific exceptions if possible
            logger.error(f"snapshot_download failed for {model_name} ({model_id}): {e}", exc_info=True)
            await update_download_status(model_name, "failed", f"Download error: {str(e)}")
            return False

    except Exception as e:
        # This was part of the try block, error handling moved into the try-except for snapshot_download
        pass # Error handling is now more specific within the try block

async def run_download_task(models_to_download: Dict[str, Any], output_dir: str, token: Optional[str], force: bool):
    """Async background task function to download multiple models."""
    logger.info(f"Async background download task started for {len(models_to_download)} models.")
    success_count = 0
    total_models = len(models_to_download)
    failed_models = []

    for model_name, config in models_to_download.items():
        logger.info(f"[{success_count+len(failed_models)+1}/{total_models}] Processing model in background: {model_name}")
        try:
            if await _download_single_model(model_name, config, output_dir, token, force):
                await update_download_status(model_name, "completed", "Download successful.")
                success_count += 1
            else:
                # _download_single_model already updates status to "failed" with a message if it returns False
                failed_models.append(model_name)
        except Exception as e:
            logger.error(f"Unhandled exception downloading {model_name}: {e}", exc_info=True)
            await update_download_status(model_name, "failed", f"Unexpected error: {str(e)}")
            failed_models.append(model_name)

    logger.info(f"Background download task finished.")
    logger.info(f"Successfully downloaded: {success_count}/{total_models}")
    if failed_models:
        logger.warning(f"Failed to download: {len(failed_models)}/{total_models} models: {', '.join(failed_models)}")
    # SSE will handle notifications based on download_statuses updates

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

# --- SSE Endpoint for Download Status ---
@router.get("/models/download/stream-status")
async def stream_download_status(request: Request):
    """
    Streams download status updates using Server-Sent Events (SSE).
    Sends the current status of all tracked downloads immediately upon connection,
    and then sends updates whenever a download status changes.
    """
    async def event_generator():
        global download_statuses
        global status_event
        
        # Send current state immediately
        # Make a deep copy to avoid issues if download_statuses is modified during serialization
        initial_statuses = {k: v.copy() for k, v in download_statuses.items()}
        if initial_statuses:
            try:
                yield f"data: {json.dumps(initial_statuses)}\n\n"
            except Exception as e:
                logger.error(f"Error serializing initial download_statuses for SSE: {e}")

        while True:
            if await request.is_disconnected():
                logger.info("SSE client disconnected for download status.")
                break
            try:
                # Wait for the event to be set, with a timeout to periodically check for disconnect
                await asyncio.wait_for(status_event.wait(), timeout=1.0)
                status_event.clear() # Clear the event once handled

                # Send all statuses if any change occurred, or send specific changes
                # For simplicity, sending all statuses on any change.
                # A more optimized version might send only deltas.
                current_statuses_snapshot = {k: v.copy() for k, v in download_statuses.items()}
                if current_statuses_snapshot:
                    try:
                        yield f"data: {json.dumps(current_statuses_snapshot)}\n\n"
                    except Exception as e:
                        logger.error(f"Error serializing download_statuses for SSE: {e}")
            
            except asyncio.TimeoutError:
                # Timeout is expected, just means no new event, loop to check disconnect or wait again
                pass
            except asyncio.CancelledError:
                logger.info("SSE event_generator task cancelled.")
                break # Exit loop on cancellation
            except Exception as e:
                logger.error(f"Error in SSE event_generator: {e}", exc_info=True)
                try:
                    yield f"data: {json.dumps({'error': 'SSE stream error', 'detail': str(e)})}\n\n"
                except Exception: pass # Ignore if can't send error
                break # Stop streaming on unhandled errors
        logger.info("SSE event_generator for download status finished.")

    return EventSourceResponse(event_generator(), media_type="text/event-stream")