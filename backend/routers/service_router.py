import logging
import time
import os
import requests # Added for API test
from fastapi import APIRouter, HTTPException, status as http_status, Path
from typing import Optional, Any # Added for API test response

# Import models, config, and utils using relative paths
from ..models import ServiceActionResponse, ConfiguredModelStatus, ApiTestResponse # Updated import
from ..utils.systemctl_utils import run_systemctl_command
# Import config helpers including new active model functions
from ..config import load_model_config, read_active_model_key, write_active_model_key
from .models_router import get_available_models_internal # Import helper from models router

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/service", # Prefix all routes in this router with /service
    tags=["Service Management"] # Tag for OpenAPI docs
)

# WARNING: These endpoints execute sudo commands.
# Ensure the user running the FastAPI server has *specific, limited* passwordless sudo rights
# ONLY for 'systemctl start/stop/restart/enable/disable vllm'.

@router.post("/start", response_model=ServiceActionResponse, summary="Start vLLM Service")
async def start_service():
    """Starts the vLLM systemd service."""
    return run_systemctl_command("start")

@router.post("/stop", response_model=ServiceActionResponse, summary="Stop vLLM Service")
async def stop_service():
    """Stops the vLLM systemd service."""
    return run_systemctl_command("stop")

@router.post("/activate/{model_key}", response_model=ServiceActionResponse, summary="Activate Model & Restart Service")
async def activate_model_and_restart_service(
    model_key: str = Path(..., description="The configuration key of the model to activate (e.g., 'llama_3_8b_instruct').")
):
    """
    Sets the specified model as active and restarts the vLLM systemd service.
    The service (via vllm_launcher.py) will read the active model key on startup
    and use its specific configuration (e.g., tensor_parallel_size).
    Requires the target model to be configured and preferably downloaded.
    """
    logger.info(f"Received request to activate model: {model_key}")

    # 1. Validate model key exists in config
    model_configs = load_model_config()
    if model_key not in model_configs:
        logger.error(f"Activation failed: Model key '{model_key}' not found in configuration.")
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Model key '{model_key}' not found in configuration."
        )

    # 2. Write the active model key to the file using config helper
    try:
        write_active_model_key(model_key) # Use the centralized function
    except IOError as e: # Catch potential IOErrors from the helper
        logger.error(f"Failed to write active model key: {e}")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set active model configuration: {e}"
        )

    # 3. Restart the service using the utility function
    logger.info(f"Triggering service restart to activate model '{model_key}'...")
    response = run_systemctl_command("restart") # This already handles exceptions

    # Add a small delay to allow service to potentially start/settle
    time.sleep(3)
    # Append activation info to the response message
    response.message = f"Activation successful for '{model_key}'. {response.message}"
    return response


@router.post("/enable", response_model=ServiceActionResponse, summary="Enable vLLM Service at Boot")
async def enable_service():
    """Enables the vLLM systemd service to start automatically on system boot."""
    return run_systemctl_command("enable")

@router.post("/disable", response_model=ServiceActionResponse, summary="Disable vLLM Service at Boot")
async def disable_service():
    """Disables the vLLM systemd service from starting automatically on system boot."""
    return run_systemctl_command("disable")

@router.get("/status", response_model=ConfiguredModelStatus, summary="Get vLLM Service Status")
async def get_service_status_detailed():
    """
    Returns the current status (active/inactive) and enabled status (enabled/disabled)
    of the vLLM systemd service, the currently active model key, and the list of
    all configured models.
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

    # Read the currently configured active model using config helper
    active_model_key = read_active_model_key() # Returns None if not found/error

    # Get configured models using the helper from models_router
    configured_models = get_available_models_internal()

    # Return using the Pydantic model
    return ConfiguredModelStatus(
        service_status=service_status,
        service_enabled=service_enabled,
        active_model_key=active_model_key, # Pass None or the key
        configured_models=configured_models
    )

# --- API Test Endpoint ---
@router.get("/test-vllm-api", response_model=ApiTestResponse, summary="Test vLLM API Reachability")
async def test_vllm_api():
     """Checks if the vLLM OpenAI-compatible API endpoint (/v1/models) is reachable."""
     # TODO: Make the vLLM API URL configurable via config.py?
     vllm_api_base_url = os.environ.get("VLLM_API_URL", "http://localhost:8000") # Get base URL
     vllm_models_url = f"{vllm_api_base_url.rstrip('/')}/v1/models"

     logger.info(f"Testing vLLM API endpoint: {vllm_models_url}")
     try:
          response = requests.get(vllm_models_url, timeout=5)
          if response.status_code == 200:
               logger.info("vLLM API test successful.")
               return ApiTestResponse(
                    status="ok",
                    message="vLLM API is reachable and returned model list.",
                    vllm_api_reachable=True,
                    vllm_response=response.json()
               )
          else:
               logger.warning(f"vLLM API test failed: Received status code {response.status_code}")
               return ApiTestResponse(
                    status="error",
                    message=f"vLLM API returned non-200 status: {response.status_code}",
                    vllm_api_reachable=False,
                    error_details=response.text[:500]
               )
     except requests.exceptions.Timeout:
          logger.error("vLLM API test failed: Connection timed out.")
          return ApiTestResponse(
               status="error", message="Connection to vLLM API timed out.", vllm_api_reachable=False, error_details="Timeout"
          )
     except requests.exceptions.RequestException as e:
          logger.error(f"vLLM API test failed: Connection error: {e}")
          return ApiTestResponse(
               status="error", message="Failed to connect to vLLM API.", vllm_api_reachable=False, error_details=str(e)
          )
     except Exception as e:
          logger.error(f"vLLM API test failed: Unexpected error: {e}")
          return ApiTestResponse(
               status="error", message="An unexpected error occurred during API test.", vllm_api_reachable=False, error_details=str(e)
          )