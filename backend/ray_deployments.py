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

# --- Ray Serve Deployment Builder for Dynamic Model Serving ---
from ray.serve.llm import build_llm_deployment
# Removed unused import: from ray.serve.config import ScalingConfig

def build_llm_deployments(full_config: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
    """
    Build a dictionary of Ray Serve Deployments for OpenAI-compatible serving
    based on the provided configuration using `build_llm_deployment`.

    Args:
        full_config: The complete model configuration dictionary loaded from model_config.json.

    Returns:
        A dictionary mapping model keys (used as route prefixes) to their
        corresponding Ray Serve Deployment objects, or None if no models are ready.
    """
    deployments = {}
    models_prepared_count = 0

    logger.info("Building Ray Serve Deployments based on 'serve: true' flag and download status...")
    for key, conf in full_config.items():
        if not isinstance(conf, dict):
            logger.warning(f"Skipping invalid config entry for key '{key}': not a dictionary.")
            continue

        serve_flag = conf.get("serve", False)
        model_path = os.path.join(MODELS_DIR, key)
        is_downloaded = os.path.isdir(model_path) and bool(os.listdir(model_path))

        if serve_flag and is_downloaded:
            logger.info(f"Preparing deployment for model '{key}'.")
            try:
                # Prepare engine config for build_llm_deployment
                engine_conf = {
                    "tensor_parallel_size": conf.get("tensor_parallel_size", 1),
                    "max_model_len": conf.get("max_model_len"),
                    "dtype": conf.get("dtype", "auto"),
                    "gpu_memory_utilization": conf.get("gpu_memory_utilization", 0.90),
                    # Add other relevant engine args from config if needed
                    # e.g., "quantization": conf.get("quantization")
                }
                # Filter out None values as build_llm_deployment might expect concrete values or defaults
                engine_conf = {k: v for k, v in engine_conf.items() if v is not None}

                # Prepare scaling config (optional, can be customized)
                # Example: Use num_workers based on tensor_parallel_size?
                # Or let Ray auto-scale? For simplicity, start without explicit scaling config.
                # scaling_conf = ScalingConfig(num_workers=conf.get("tensor_parallel_size", 1))

                # Build the deployment for this specific model
                # Note: The parameter name might be model_id or model_path or path depending on the Ray version
                try:
                    # Try with model_id first (newer Ray versions)
                    deployment = build_llm_deployment(
                        model_id=model_path, # Use the local path
                        engine_config=engine_conf,
                        # scaling_config=scaling_conf, # Optional
                        # Add other build_llm_deployment args if needed from conf
                    )
                except TypeError as e:
                    if "unexpected keyword argument 'model_id'" in str(e):
                        try:
                            # Try with model_path (some Ray versions)
                            deployment = build_llm_deployment(
                                model_path=model_path, # Use the local path
                                engine_config=engine_conf,
                                # scaling_config=scaling_conf, # Optional
                                # Add other build_llm_deployment args if needed from conf
                            )
                        except TypeError as e2:
                            if "unexpected keyword argument 'model_path'" in str(e2):
                                try:
                                    # Try with just path (oldest Ray versions)
                                    deployment = build_llm_deployment(
                                        path=model_path, # Use the local path
                                        engine_config=engine_conf,
                                        # scaling_config=scaling_conf, # Optional
                                        # Add other build_llm_deployment args if needed from conf
                                    )
                                except TypeError as e3:
                                    # Last resort for tests - try with positional argument
                                    try:
                                        deployment = build_llm_deployment(
                                            model_path,  # Positional argument
                                            engine_config=engine_conf,
                                        )
                                    except Exception as e4:
                                        logger.error(f"All parameter name attempts failed for model '{model_name}': {e4}")
                                        raise
                            else:
                                # Re-raise if it's a different TypeError
                                raise e2
                    else:
                        # Re-raise if it's a different TypeError
                        raise
                # Use the model key as the route prefix
                deployments[f"/{key}"] = deployment # Add leading slash for clarity? Ray might add it anyway.
                models_prepared_count += 1
                logger.info(f"Successfully prepared deployment for model '{key}'.")
            except Exception as e:
                logger.error(f"Failed to build deployment for model '{key}': {e}", exc_info=True)

        elif serve_flag and not is_downloaded:
             logger.warning(f"Model '{key}' marked to serve but not downloaded. Skipping deployment.")
        # else: # Model not marked to serve

    if models_prepared_count == 0:
        logger.warning("No models configured, marked to serve, AND downloaded found. Returning None (no deployments to run).")
        return None

    logger.info(f"Prepared {models_prepared_count} model deployment(s).")
    return deployments

# Renamed function build_llm_app to build_llm_deployments to reflect its new return type

# --- Commented out old code ---
# ...