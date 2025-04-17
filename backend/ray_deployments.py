import os
from typing import Dict, Any, Optional
import logging

# Import the vLLM engine integration for Ray Serve
# Ensure vllm is installed in the same environment Ray Serve runs in
try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    logging.error("vLLM library not found. Cannot define Ray Serve deployment for vLLM.")
    VLLM_AVAILABLE = False
    # Define dummy classes or raise error to prevent deployment attempt?
    class AsyncEngineArgs: pass
    class AsyncLLMEngine: pass

# Import config to get paths (adjust relative import as needed)
try:
    from .config import VLLM_HOME, MODELS_DIR
except ImportError:
     # Fallback if run directly or path issues
     VLLM_HOME = os.environ.get("VLLM_HOME", "/opt/vllm")
     MODELS_DIR = os.path.join(VLLM_HOME, "models")

# Configure logger for this module
logger = logging.getLogger("ray.serve") # Use Ray Serve's logger

# --- Ray Serve LLMApp Builder for Dynamic Model Serving ---

def build_llm_app(full_config: Dict[str, Dict]) -> Optional[Any]: # Return Any as LLMApp might not be importable here
    """
    Build a Ray Serve LLMApp configuration dictionary for OpenAI-compatible serving
    based on the provided configuration.

    Args:
        full_config: The complete model configuration dictionary loaded from model_config.json.

    Returns:
        An LLMApp instance configured with models marked "serve: true" and downloaded,
        or None if no models are ready to be served or LLMApp cannot be imported.
    """
    # Attempt to import LLMApp directly inside the function.
    # Removed the try...except block to let the ImportError propagate if it occurs.
    from ray.serve.llm import LLMApp

    app_config = {}
    models_prepared_count = 0

    logger.info("Building LLMApp config based on 'serve: true' flag and download status...")
    for key, conf in full_config.items():
        if not isinstance(conf, dict):
            logger.warning(f"Skipping invalid config entry for key '{key}': not a dictionary.")
            continue

        serve_flag = conf.get("serve", False)
        model_path = os.path.join(MODELS_DIR, key)
        is_downloaded = os.path.isdir(model_path) and bool(os.listdir(model_path))

        if serve_flag and is_downloaded:
            logger.info(f"Adding model '{key}' to LLMApp config.")
            app_config[key] = {
                "model": model_path, # Use the local path to the downloaded model directory
                "engine_config": {
                    "tensor_parallel_size": conf.get("tensor_parallel_size", 1),
                    "max_model_len": conf.get("max_model_len"), # Let vLLM read from model's internal config if None?
                    "dtype": conf.get("dtype", "auto"),
                    "gpu_memory_utilization": conf.get("gpu_memory_utilization", 0.90), # Allow override from config
                    # Add other relevant engine args from config if needed
                    # e.g., "quantization": conf.get("quantization")
                },
                # Provider defaults to OpenAIProvider, no need to specify usually
                # Route prefix defaults to model key, also usually fine
            }
            models_prepared_count += 1
        elif serve_flag and not is_downloaded:
             logger.warning(f"Model '{key}' marked to serve but not downloaded. Skipping for LLMApp.")
        # else: # Model not marked to serve, no need to log anything here

    if models_prepared_count == 0:
        logger.warning("No models configured, marked to serve, AND downloaded found. Returning None (no LLMApp to deploy).")
        return None

    logger.info(f"Prepared LLMApp config with {models_prepared_count} model(s).")
    # LLMApp takes **kwargs for models
    return LLMApp(**app_config)

# --- Commented out old code ---
# (Includes Option 1: VLLMModelServer and old build_llm_app structure)
# ... (rest of the commented code from the original file can be omitted for brevity) ...