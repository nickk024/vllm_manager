# All logs from this module are unified via the root logger in backend/main.py.
# Do not set up per-module file handlers or basicConfig here.
import logging
import time
import os
import requests
from fastapi import APIRouter, HTTPException, status as http_status, Path
from typing import Optional, Any

# Ray Serve imports (Keep for potential future use or status checks)
# import ray
# from ray import serve
# from ray.serve.exceptions import RayServeException

# Import models, config, and utils using relative paths
from ..models import ServiceActionResponse, ConfiguredModelStatus, ApiTestResponse, ModelInfo # Updated import
# Removed systemctl_utils import as it's obsolete
from ..config import load_model_config, MODELS_DIR # Removed SERVICE_NAME, No longer need ACTIVE_MODEL_FILE functions
from .models_router import get_configured_models_internal # Import helper from models router

logger = logging.getLogger(__name__)
router = APIRouter(
    # prefix="/service", # Removed redundant prefix, it's handled in main.py
    tags=["Service Management (Ray Serve)"]
)

# Note: This router now manages Ray Serve status and API health.
# Systemd and vllm_launcher.py logic has been removed.

# TODO: Add endpoints for Ray Serve deployment management (e.g., manual model unload, reload, etc.)

# Systemd service management endpoints removed.

@router.get("/status", response_model=ConfiguredModelStatus, summary="Get Production Health Status")
async def get_ray_serve_status():
    """
    Production health check endpoint with dependency verification and cluster status.
    Returns detailed service health including Ray cluster status and critical dependencies.
    """
    try:
        # Verify critical dependencies first
        import pyarrow  # noqa: F401
        import ray
        from ray import serve
        # from ray.serve.llm import LLMApp  # noqa: F401 <-- Removed unused import

        # Get detailed Ray status
        ray_status = {
            "initialized": ray.is_initialized(),
            "nodes": len(ray.nodes()) if ray.is_initialized() else 0,
            "resources": ray.available_resources() if ray.is_initialized() else {},
            "version": ray.__version__
        }
        
        # Get basic Serve status
        serve_status = "unknown"
        if ray.is_initialized():
            ray_status = "running"
            try:
                serve_controller = serve.api._get_global_client()
                serve_status = "running" if serve_controller else "not_running"
            except Exception:
                serve_status = "not_running"
        else:
            ray_status = "not_running"
            serve_status = "not_running"
    except Exception as e:
        logger.error(f"Error checking Ray Serve status: {e}")
        ray_status = "error"
        serve_status = "error"

    # Get configured models using the helper from models_router
    configured_models = get_configured_models_internal() # This now includes serve_status

    # Return the updated status model
    return ConfiguredModelStatus(
        ray_serve_status=f"Ray: {ray_status}, Serve: {serve_status}",
        configured_models=configured_models # This list now includes the 'serve' status for each model
    )

# Keep API test endpoint
@router.get("/test-vllm-api", response_model=ApiTestResponse, summary="Test vLLM API Reachability")
async def test_vllm_api():
     """
     Checks if the vLLM OpenAI-compatible API endpoint (/v1/models) is reachable.
     Assumes vLLM runs on port 8000 as started by the launcher.
     """
     vllm_api_base_url = os.environ.get("VLLM_API_URL", "http://localhost:8000")
     vllm_models_url = f"{vllm_api_base_url.rstrip('/')}/v1/models"

     logger.info(f"Testing vLLM API endpoint: {vllm_models_url}")
     try:
          response = requests.get(vllm_models_url, timeout=10)
          if response.status_code == 200:
               logger.info("vLLM API test successful.")
               return ApiTestResponse(
                    status="ok", message="vLLM API is reachable and returned model list.",
                    vllm_api_reachable=True, vllm_response=response.json()
               )
          else:
               logger.warning(f"vLLM API test failed: Received status code {response.status_code}")
               return ApiTestResponse(
                    status="error", message=f"vLLM API returned non-200 status: {response.status_code}",
                    vllm_api_reachable=False, error_details=response.text[:500]
               )
     except requests.exceptions.Timeout:
          logger.error("vLLM API test failed: Connection timed out.")
          return ApiTestResponse(status="error", message="Connection to vLLM API timed out.", vllm_api_reachable=False, error_details="Timeout")
     except requests.exceptions.RequestException as e:
          logger.error(f"vLLM API test failed: Connection error: {e}")
          return ApiTestResponse(status="error", message="Failed to connect to vLLM API.", vllm_api_reachable=False, error_details=str(e))
     except Exception as e:
          logger.error(f"vLLM API test failed: Unexpected error: {e}")
          return ApiTestResponse(status="error", message="An unexpected error occurred during API test.", vllm_api_reachable=False, error_details=str(e))