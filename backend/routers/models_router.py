# All logs from this module are unified via the root logger in backend/main.py.
# Do not set up per-module file handlers or basicConfig here.
import os
import logging
from fastapi import APIRouter, HTTPException, status as http_status, Query, Path, Body
# Removed duplicate import line below
from typing import List, Dict, Any, Optional

# Import models, config, and utils using relative paths
from ..models import ModelInfo, PopularModelInfo, AddModelRequest, GeneralResponse, ToggleServeRequest # Changed ConfiguredModelInfo to ModelInfo
from ..config import load_model_config, save_model_config, CURATED_MODELS_DATA, MODELS_DIR
from ..utils.gpu_utils import get_gpu_count
from ..utils.hf_utils import fetch_dynamic_popular_models

logger = logging.getLogger(__name__)
router = APIRouter() # Management endpoints will be prefixed in main.py

# --- Helper ---
def get_configured_models_internal() -> List[ModelInfo]:
    """Internal helper to get configured models and their status."""
    config = load_model_config()
    models = []
    for key, info in config.items():
        model_dir = os.path.join(MODELS_DIR, key)
        downloaded = os.path.isdir(model_dir) and bool(os.listdir(model_dir))
        # Ensure info is a dict before accessing .get()
        if isinstance(info, dict):
            serve_status = info.get("serve", False) # Default serve status to False if missing
            model_id = info.get("model_id", "N/A")
        else:
            # Handle case where config entry isn't a dictionary (corrupted?)
            serve_status = False
            model_id = "Invalid Config Entry"
            logger.warning(f"Config entry for key '{key}' is not a dictionary: {info}")

        # Note: ModelInfo has 'name', not 'config_key'. We map key to name.
        # Populate the ModelInfo object, including the serve status
        models.append(ModelInfo(
            name=key,
            model_id=model_id,
            downloaded=downloaded,
            serve=serve_status # Add the serve status here
        ))
    return models

# --- Routes ---
@router.get("/models", response_model=List[ModelInfo], summary="List Configured Models")
async def list_configured_models():
    """
    List all models currently present in the model_config.json file,
    indicating download status and whether they are marked to be served.
    """
    return get_configured_models_internal()

@router.get("/models/popular", response_model=List[PopularModelInfo], summary="List Popular Models (Filtered by VRAM)")
async def list_popular_models_filtered(
    available_vram_gb: Optional[float] = Query(None, description="Available system VRAM in GB to filter models."),
    top_n: int = Query(10, description="Maximum number of models to return."),
    hf_token: Optional[str] = Query(None, description="Hugging Face token (optional).")
):
    """
    List popular text-generation models from Hugging Face Hub,
    with VRAM estimation and filtering. Returns top N models estimated to fit.
    """
    logger.info(f"Request received for popular models. Available VRAM filter: {available_vram_gb} GB, Top N: {top_n}")
    models = fetch_dynamic_popular_models(
        available_vram_gb=available_vram_gb, top_n=top_n, hf_token=hf_token
    )
    response_models = []
    for model_data in models:
         cleaned_data = {k: v for k, v in model_data.items() if not k.startswith('_')}
         try:
              response_models.append(PopularModelInfo(**cleaned_data))
         except Exception as e:
              logger.error(f"Pydantic validation failed for popular model data {cleaned_data}: {e}")
              continue
    return response_models


@router.post("/config/models", response_model=GeneralResponse, status_code=http_status.HTTP_201_CREATED, summary="Add Model to Configuration")
async def add_model_to_config(req: AddModelRequest):
    """
    Adds a model (specified by model_id) to the model_config.json using
    default settings and sets 'serve: false'. Does not download the model.
    Returns the generated config key on success. Requires a service redeploy
    if you later mark this model to be served.
    """
    current_config = load_model_config()
    added_models_info = {}
    skipped_count = 0
    gpu_count = get_gpu_count() # Fetch GPU count once

    for model_id_to_add in req.model_ids:
        existing_key = next((key for key, conf in current_config.items() if isinstance(conf, dict) and conf.get("model_id") == model_id_to_add), None)
        if existing_key:
            logger.info(f"Model ID '{model_id_to_add}' already exists in config under key '{existing_key}'. Skipping.")
            skipped_count += 1
            continue

        config_key = model_id_to_add.lower().replace("/", "_").replace("-", "_").replace(".", "_")
        original_config_key = config_key
        counter = 1
        while config_key in current_config:
            config_key = f"{original_config_key}_{counter}"
            counter += 1

        # Determine tensor_parallel_size based on gpu_count and model size
        tensor_parallel_size = 1 # Default for smaller models or single GPU
        if gpu_count == 3:
            # For 3 GPUs, use TP=3 for very large models.
            if any(s in model_id_to_add.lower() for s in ['70b', '65b', '40b', '34b', '32b']): # Keywords for large models
                tensor_parallel_size = 3
            elif any(s in model_id_to_add.lower() for s in ['13b', '20b']): # Example for medium models
                 tensor_parallel_size = 2 # Or keep 1 if preferred to run more models concurrently
            # else TP remains 1 for smaller models
        elif gpu_count == 2:
            # For 2 GPUs, use TP=2 for large/medium models
            if any(s in model_id_to_add.lower() for s in ['70b', '65b', '40b', '34b', '32b', '13b', '20b']):
                tensor_parallel_size = 2
        elif gpu_count >= 4: # Example for 4+ GPUs, could use TP=4 for very large models
            if any(s in model_id_to_add.lower() for s in ['70b', '65b']):
                tensor_parallel_size = min(4, gpu_count) # Use up to 4, or all if more
            elif any(s in model_id_to_add.lower() for s in ['40b', '34b', '32b']):
                tensor_parallel_size = min(2, gpu_count) # TP=2 for these on 4+ GPUs
        # For gpu_count == 1, TP remains 1.
        # This logic prioritizes utilizing available GPUs for larger models.

        # Add serve: false by default
        new_config_entry = {
            "model_id": model_id_to_add,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": 4096,
            "dtype": "bfloat16",
            "serve": False # Default to not serving
        }

        current_config[config_key] = new_config_entry
        added_models_info[config_key] = model_id_to_add
        logger.info(f"Added model '{config_key}' ({model_id_to_add}) to config with serve=False, TP={tensor_parallel_size}.")

    if added_models_info:
        save_model_config(current_config)
        return GeneralResponse(
             status="ok",
             message=f"Configuration updated. Added {len(added_models_info)} model(s) (marked as not served), Skipped {skipped_count}.",
             details={"added_keys": list(added_models_info.keys())}
        )
    else:
         return GeneralResponse(
              status="skipped",
              message=f"No models added. Skipped {skipped_count} (already configured)."
         )

from ..ray_deployments import build_llm_deployments # Renamed function

@router.put("/config/models/{model_key}/serve", response_model=GeneralResponse, summary="Toggle Serve Status for a Model")
async def toggle_model_serve_status(
    model_key: str = Path(..., description="The configuration key of the model to modify."),
    req: ToggleServeRequest = Body(...)
):
    """
    Sets the 'serve' status for a specific model in the configuration file.
    Triggers a Ray Serve redeployment for the change to take effect.
    """
    logger.info(f"Request to set serve status for '{model_key}' to {req.serve}")
    current_config = load_model_config()

    if model_key not in current_config:
        logger.error(f"Model key '{model_key}' not found in configuration.")
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Model key '{model_key}' not found.")

    if not isinstance(current_config[model_key], dict):
         logger.error(f"Invalid config format for model key '{model_key}'.")
         raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid configuration for model '{model_key}'.")

    current_config[model_key]["serve"] = req.serve
    save_model_config(current_config)

    # Trigger Ray Serve redeployment
    try:
        import ray
        from ray import serve
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
        # Build new dictionary of deployments and redeploy
        llm_deployments = build_llm_deployments(current_config)
        serve.run(
            llm_deployments, # Pass the dictionary of deployments
            # name="vllm_app", # Name is less relevant for dict deployment
            # route_prefix="/", # Prefixes are defined by dict keys
            host="0.0.0.0",
            port=8000,
            blocking=False
        )
        logger.info("Ray Serve redeployment triggered after serve status change.")
        status_str = "enabled" if req.serve else "disabled"
        return GeneralResponse(
            status="ok",
            message=f"Serve status for model '{model_key}' set to {status_str}. Ray Serve redeployed."
        )
    except Exception as e:
        logger.error(f"Failed to redeploy Ray Serve after serve status change: {e}", exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Serve status updated, but failed to redeploy Ray Serve: {e}"
        )

# TODO: Add endpoint to REMOVE a model from config? DELETE /config/models/{model_key}

@router.delete("/config/models/{model_key}", response_model=GeneralResponse, summary="Remove Model from Configuration")
async def remove_model_from_config(
    model_key: str = Path(..., description="The configuration key of the model to remove.")
):
    """
    Removes a model entry from the model_config.json file.
    If the model is currently marked as 'serve: true', its status will be set to 'serve: false'
    and a Ray Serve redeployment will be triggered to unload it.
    This does not delete the downloaded model files from disk.
    """
    logger.info(f"Request to remove model '{model_key}' from configuration.")
    current_config = load_model_config()

    if model_key not in current_config:
        logger.warning(f"Model key '{model_key}' not found in configuration. Cannot remove.")
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Model key '{model_key}' not found in configuration.")

    model_data = current_config.get(model_key, {})
    was_serving = False
    if isinstance(model_data, dict) and model_data.get("serve", False):
        logger.info(f"Model '{model_key}' is currently marked as 'serve: true'. It will be unserved before removal from config.")
        current_config[model_key]["serve"] = False # Mark as not serving
        was_serving = True
        # Note: We will save the config after deletion, then redeploy if it was serving.

    del current_config[model_key]
    save_model_config(current_config)
    logger.info(f"Model '{model_key}' removed from configuration file.")

    if was_serving:
        logger.info(f"Model '{model_key}' was serving. Triggering Ray Serve redeployment to unload it.")
        try:
            import ray
            from ray import serve
            if not ray.is_initialized():
                # This case should ideally not happen if the app is running, but as a safeguard:
                logger.warning("Ray not initialized during model removal. Assuming local setup for re-init.")
                ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
            
            # Re-build deployments based on the *new* config (where the model is now removed or serve=false)
            # The build_llm_deployments function will naturally exclude the removed model or models with serve=false
            llm_deployments = build_llm_deployments(current_config) # current_config is now without the deleted model
            
            serve.run(
                llm_deployments if llm_deployments else {}, # Pass empty dict if no models left to serve
                host="0.0.0.0",
                port=8000,
                blocking=False
            )
            logger.info(f"Ray Serve redeployment triggered to reflect removal/unserving of '{model_key}'.")
            return GeneralResponse(
                status="ok",
                message=f"Model '{model_key}' removed from config and Ray Serve redeployed to unload it."
            )
        except Exception as e:
            logger.error(f"Model '{model_key}' removed from config, but failed to redeploy Ray Serve: {e}", exc_info=True)
            # The model is removed from config, but might still be running in Serve if redeploy failed.
            # This is a potentially inconsistent state.
            return GeneralResponse(
                status="partial_error",
                message=f"Model '{model_key}' removed from config, but Ray Serve redeploy failed: {str(e)}. Manual check of Ray Serve may be needed."
            )
            
    return GeneralResponse(
        status="ok",
        message=f"Model '{model_key}' successfully removed from configuration."
    )