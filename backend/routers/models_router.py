import os
import logging
from fastapi import APIRouter, HTTPException, status as http_status, Query
from typing import List, Dict, Any, Optional

# Import models, config, and utils using relative paths
from ..models import ModelInfo, PopularModelInfo, AddModelRequest
from ..config import load_model_config, save_model_config, CURATED_MODELS_DATA, MODELS_DIR
from ..utils.gpu_utils import get_gpu_count
from ..utils.hf_utils import fetch_dynamic_popular_models # Import the new HF util

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper ---
def get_available_models_internal() -> List[ModelInfo]:
    """Internal helper to get configured models and their status."""
    config = load_model_config()
    models = []
    for name, info in config.items():
        model_dir = os.path.join(MODELS_DIR, name)
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

@router.get("/models/popular", response_model=List[PopularModelInfo], summary="List Popular Models (Filtered by VRAM)")
async def list_popular_models_filtered(
    available_vram_gb: Optional[float] = Query(None, description="Available system VRAM in GB to filter models."),
    top_n: int = Query(10, description="Maximum number of models to return."),
    hf_token: Optional[str] = Query(None, description="Hugging Face token for potentially fetching more models or private ones (optional).")
):
    """
    List popular text-generation models from Hugging Face Hub,
    with a very rough VRAM estimation and filtering based on available VRAM.
    Returns top N models estimated to fit.
    """
    logger.info(f"Request received for popular models. Available VRAM filter: {available_vram_gb} GB, Top N: {top_n}")
    # Use the new utility function to fetch and filter
    models = fetch_dynamic_popular_models(
        available_vram_gb=available_vram_gb,
        top_n=top_n,
        hf_token=hf_token
        # Can add limit parameter here if needed
    )
    # Convert the result to the Pydantic model format expected by the API response
    # Note: fetch_dynamic_popular_models already returns a list of dicts matching PopularModelInfo structure
    # If the structure differed, mapping would be needed here.
    # We might want to remove the _debug fields before returning
    response_models = []
    for model_data in models:
         # Create Pydantic model instance, excluding debug fields if necessary
         response_models.append(PopularModelInfo(**{k: v for k, v in model_data.items() if not k.startswith('_')}))

    return response_models


@router.post("/config/models", status_code=http_status.HTTP_201_CREATED, summary="Add Models to Configuration")
async def add_models_to_config(req: AddModelRequest):
    """
    Add selected popular models (by model_id) to the model_config.json.
    Uses the CURATED_MODELS_DATA list to find model details.
    This updates the configuration file but does not download the models.
    Use the /models/download endpoint to download after adding.
    """
    # NOTE: This currently only uses CURATED_MODELS_DATA.
    # To add models fetched dynamically via /models/popular, this endpoint
    # would need access to the fetched data or require the client to provide
    # the necessary config details along with the model_id.
    # For now, it only works reliably for models in the hardcoded CURATED_MODELS_DATA list.
    # TODO: Refactor this to handle dynamically fetched models better.

    current_config = load_model_config()
    added_count = 0
    skipped_count = 0
    models_added_keys = []
    gpu_count = get_gpu_count()

    # Use CURATED list for lookup for now
    popular_map = {m["model_id"]: m for m in CURATED_MODELS_DATA}

    for model_id_to_add in req.model_ids:
        if model_id_to_add not in popular_map:
            logger.warning(f"Requested model_id '{model_id_to_add}' not found in curated popular list. Skipping add to config.")
            # If dynamic fetching was implemented, we'd check that list too.
            skipped_count += 1
            continue

        popular_model_data = popular_map[model_id_to_add]
        config_key = popular_model_data["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        original_config_key = config_key
        counter = 1
        while config_key in current_config:
             existing_entry = next((item for item in current_config.values() if item.get("model_id") == model_id_to_add), None)
             if existing_entry:
                  logger.info(f"Model ID '{model_id_to_add}' already exists in config under key '{next(k for k,v in current_config.items() if v==existing_entry)}'. Skipping.")
                  skipped_count += 1
                  config_key = None
                  break
             config_key = f"{original_config_key}_{counter}"
             counter += 1

        if config_key is None: continue

        base_config = popular_model_data.get("config", {})
        model_size_gb = popular_model_data.get("size_gb", 0)
        tensor_parallel_size = base_config.get("tensor_parallel_size", 1)
        if model_size_gb > 30 and gpu_count > 1: tensor_parallel_size = min(tensor_parallel_size, gpu_count)
        elif model_size_gb <= 10 and gpu_count >= 1 : tensor_parallel_size = 1
        elif gpu_count == 0: logger.warning(f"No GPUs detected, setting tensor_parallel_size=1 for {config_key}"); tensor_parallel_size = 1

        current_config[config_key] = {
            "model_id": model_id_to_add,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": base_config.get("max_model_len", 4096),
            "dtype": base_config.get("dtype", "bfloat16")
        }
        added_count += 1
        models_added_keys.append(config_key)
        logger.info(f"Added model '{config_key}' ({model_id_to_add}) to config.")

    if added_count > 0:
        save_model_config(current_config)

    return {
        "status": "ok",
        "message": f"Configuration updated. Added: {added_count}, Skipped: {skipped_count}.",
        "added_model_keys": models_added_keys
    }