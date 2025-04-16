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
from ..models import ServiceActionResponse, ServiceStatusResponse, ApiTestResponse, ConfiguredModelInfo # Updated import
from ..utils.systemctl_utils import run_systemctl_command # Keep for start/stop/enable/disable/restart
from ..config import load_model_config, MODELS_DIR, SERVICE_NAME # No longer need ACTIVE_MODEL_FILE functions
from .models_router import get_configured_models_internal # Import helper from models router

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/service", # Prefix for management API
    tags=["Service Management (Systemd)"] # Updated tag
)

# Note: This router now controls the systemd service which runs vllm_launcher.py
# The launcher itself handles loading multiple models based on config.

# WARNING: These endpoints execute sudo commands if not run as root.
# Ensure the user running the FastAPI server has *specific, limited* passwordless sudo rights
# ONLY for 'systemctl start/stop/restart/enable/disable vllm'.

@router.post("/start", response_model=ServiceActionResponse, summary="Start vLLM Service")
async def start_service():
    """
    Starts the vLLM systemd service.
    The launcher script will attempt to load models marked 'serve: true'.
    """
    return run_systemctl_command("start")

@router.post("/stop", response_model=ServiceActionResponse, summary="Stop vLLM Service")
async def stop_service():
    """Stops the vLLM systemd service."""
    return run_systemctl_command("stop")

@router.post("/restart", response_model=ServiceActionResponse, summary="Restart vLLM Service")
async def restart_service():
    """
    Restarts the vLLM systemd service.
    Use this to apply changes after modifying which models are marked 'serve: true'
    in the configuration or after downloading new models marked to serve.
    """
    response = run_systemctl_command("restart")
    # Add a small delay to allow service to potentially start/settle
    time.sleep(5) # Increased delay slightly for multi-model load
    return response

# Removed /activate/{model_key} endpoint

@router.post("/enable", response_model=ServiceActionResponse, summary="Enable vLLM Service at Boot")
async def enable_service():
    """Enables the vLLM systemd service to start automatically on system boot."""
    return run_systemctl_command("enable")

@router.post("/disable", response_model=ServiceActionResponse, summary="Disable vLLM Service at Boot")
async def disable_service():
    """Disables the vLLM systemd service from starting automatically on system boot."""
    return run_systemctl_command("disable")

@router.get("/status", response_model=ServiceStatusResponse, summary="Get vLLM Service Status")
async def get_service_status_detailed():
    """
    Returns the current status (active/inactive) and enabled status (enabled/disabled)
    of the vLLM systemd service, along with the list of all configured models
    (including their 'serve' status and download status).
    """
    service_status = "unknown"; service_enabled = "unknown"
    try:
        active_status_resp = run_systemctl_command("is-active")
        enabled_status_resp = run_systemctl_command("is-enabled")
        service_status = active_status_resp.message if active_status_resp.status == 'ok' else f"error_checking ({active_status_resp.message})"
        service_enabled = enabled_status_resp.message if enabled_status_resp.status == 'ok' else f"error_checking ({enabled_status_resp.message})"
    except HTTPException as e:
         logger.error(f"Failed to get service status via systemctl: {e.detail}")
         service_status = f"error_checking ({e.detail})"
         service_enabled = f"error_checking ({e.detail})"
    except Exception as e:
         logger.error(f"Unexpected error getting service status: {e}")
         service_status = "error_checking (unexpected)"
         service_enabled = "error_checking (unexpected)"

    # Get configured models using the helper from models_router
    configured_models = get_configured_models_internal() # This now includes serve_status

    # Return using the Pydantic model (active_model_key is removed from model)
    return ServiceStatusResponse(
        service_status=service_status,
        service_enabled=service_enabled,
        configured_models=configured_models
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