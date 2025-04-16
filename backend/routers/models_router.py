import os
import logging
from fastapi import APIRouter, HTTPException, status as http_status
from typing import List, Dict, Any

# Import models, config, and utils using relative paths
from ..models import ModelInfo, PopularModelInfo, AddModelRequest
from ..config import load_model_config, save_model_config, CURATED_MODELS_DATA, MODELS_DIR
from ..utils.gpu_utils import get_gpu_count

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper ---
def get_available_models_internal() -> List[ModelInfo]:
    """Internal helper to get configured models and their status."""
    config = load_model_config()
    models = []
    for name, info in config.items():
        model_dir = os.path.join(MODELS_DIR, name)
        # Check if directory exists and is not empty
        downloaded = os.path.isdir(model_dir) and bool(os.listdir(model_dir))
        models.append(ModelInfo(name=name, model_id=info.get("model_id", "N/A"), downloaded=downloaded))
    return models

# --- Routes ---
@router.get("/models", response_model=List[ModelInfo], summary="List Configured Models")
async def list_configured_models():
    """
    List all models currently present in the model_config.json file,
    and indicate whether their files are downloaded.
    This is the primary endpoint for OpenWebUI to discover available models.
    """
    return get_available_models_internal()

@router.get("/models/popular", response_model=List[PopularModelInfo], summary="List Popular Models")
async def list_popular_models(hf_token: str | None = None):
    """
    List popular/curated models available for configuration.
    (Currently uses a static list).
    """
    # TODO: Optionally implement fetch_popular_models from legacy script here
    # using hf_token if provided.
    return CURATED_MODELS_DATA

@router.post("/config/models", status_code=http_status.HTTP_201_CREATED, summary="Add Models to Configuration")
async def add_models_to_config(req: AddModelRequest):
    """
    Add selected popular models (by model_id) to the model_config.json.
    This updates the configuration file but does not download the models.
    Use the /models/download endpoint to download after adding.
    """
    current_config = load_model_config()
    added_count = 0
    skipped_count = 0
    models_added_keys = [] # Keep track of keys added in this request
    gpu_count = get_gpu_count() # Get GPU count for TP size adjustment

    popular_map = {m["model_id"]: m for m in CURATED_MODELS_DATA} # Map for easy lookup

    for model_id_to_add in req.model_ids:
        if model_id_to_add not in popular_map:
            logger.warning(f"Requested model_id '{model_id_to_add}' not found in popular list. Skipping.")
            skipped_count += 1
            continue

        popular_model_data = popular_map[model_id_to_add]
        # Generate a simple name for the config key (e.g., llama_3_8b_instruct)
        # Ensure uniqueness or handle potential collisions if names are similar
        config_key = popular_model_data["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        # Basic collision handling: append number if key exists
        original_config_key = config_key
        counter = 1
        while config_key in current_config:
             # Only skip if the *exact same model_id* is already configured under a different key
             existing_entry = next((item for item in current_config.values() if item.get("model_id") == model_id_to_add), None)
             if existing_entry:
                  logger.info(f"Model ID '{model_id_to_add}' already exists in config under a different key. Skipping.")
                  skipped_count += 1
                  config_key = None # Mark to skip adding
                  break
             # Otherwise, it's a different model with a similar name, append counter
             config_key = f"{original_config_key}_{counter}"
             counter += 1
        
        if config_key is None: # Skipped due to existing model_id
             continue

        if config_key in current_config: # Should not happen with collision handling, but safety check
            logger.info(f"Model '{config_key}' ({model_id_to_add}) already exists in config. Skipping.")
            skipped_count += 1
            continue

        # Prepare the config entry using data from the popular list
        base_config = popular_model_data.get("config", {})
        model_size_gb = popular_model_data.get("size_gb", 0)

        # Adjust tensor parallel size based on GPU count and model size
        tensor_parallel_size = base_config.get("tensor_parallel_size", 1)
        if model_size_gb > 30 and gpu_count > 1:
             # For large models on multi-GPU, use min(default_tp, gpu_count)
             tensor_parallel_size = min(tensor_parallel_size, gpu_count)
        elif model_size_gb <= 10 and gpu_count >= 1 : # Ensure small models use 1 GPU if available
             tensor_parallel_size = 1
        elif gpu_count == 0: # No GPUs detected
             logger.warning(f"No GPUs detected, setting tensor_parallel_size=1 for {config_key}")
             tensor_parallel_size = 1
        # Keep default TP for mid-size or if gpu_count is 1

        current_config[config_key] = {
            "model_id": model_id_to_add,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": base_config.get("max_model_len", 4096),
            "dtype": base_config.get("dtype", "bfloat16")
            # Add other relevant params from base_config if needed
        }
        added_count += 1
        models_added_keys.append(config_key)
        logger.info(f"Added model '{config_key}' ({model_id_to_add}) to config.")

    if added_count > 0:
        save_model_config(current_config)

    return {
        "status": "ok",
        "message": f"Configuration updated. Added: {added_count}, Skipped: {skipped_count}.",
        "added_model_keys": models_added_keys # Return keys of newly added models
    }