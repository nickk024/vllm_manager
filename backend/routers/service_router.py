# All logs from this module are unified via the root logger in backend/main.py.
# Do not set up per-module file handlers or basicConfig here.
import logging
import time
import os
import json # Added for parsing deployment_config_json
import requests
from fastapi import APIRouter, HTTPException, status as http_status, Path
from typing import Optional, Any, List, Dict # Added List and Dict for type hinting

import ray
from ray import serve
from ray.serve.status import DeploymentStatus, ReplicaState
from ray.serve.generated.serve_pb2 import DeploymentStatusInfo
# from ray.serve.exceptions import RayServeException # Keep if specific exceptions are handled

# Import models, config, and utils using relative paths
from ..models import ServiceActionResponse, ConfiguredModelStatus, ApiTestResponse, ModelInfo # Updated import
# Removed systemctl_utils import as it's obsolete
from ..config import load_model_config, save_model_config, MODELS_DIR # Added save_model_config, Removed SERVICE_NAME, No longer need ACTIVE_MODEL_FILE functions
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
    # Attempt to get detailed deployment statuses
    detailed_deployments = []
    if serve_status == "running":
        try:
            # serve.status() returns a ServeStatus object
            # serve.get_deployment_statuses() returns a list of DeploymentStatusInfo protobufs
            # For more user-friendly data, we can process DeploymentStatusInfo
            raw_statuses = serve.get_deployment_statuses()
            for status_proto in raw_statuses:
                # Convert protobuf to a more usable dict
                # The structure of DeploymentStatusInfo might vary slightly across Ray versions.
                # This is a common way to access its fields.
                replica_states_summary = {}
                if hasattr(status_proto, 'replica_states'): # Older Ray versions
                    for state, count in status_proto.replica_states.items():
                        replica_states_summary[ReplicaState(state).name] = count
                elif hasattr(status_proto, 'deployment_status') and hasattr(status_proto.deployment_status, 'replica_states'): # Newer Ray versions
                     for state_val, count in status_proto.deployment_status.replica_states.items():
                        replica_states_summary[ReplicaState(state_val).name] = count


                detailed_deployments.append({
                    "name": status_proto.name,
                    "status": DeploymentStatus(status_proto.status).name, # Convert enum to string
                    "message": status_proto.message,
                    "replica_states": replica_states_summary,
                    "deployment_config": json.loads(status_proto.deployment_config.deployment_config_json) if status_proto.deployment_config.deployment_config_json else None,
                })
        except Exception as e:
            logger.error(f"Could not get detailed Ray Serve deployment statuses: {e}", exc_info=True)
            # Keep serve_status as "running" but indicate an issue fetching details
            serve_status = "running (details unavailable)"


    return ConfiguredModelStatus(
        ray_serve_status=f"Ray: {ray_status}, Serve: {serve_status}", # This part remains for the general status
        configured_models=configured_models
    )

@router.get("/ray/deployments", response_model=List[Dict[str, Any]], summary="Get Detailed Ray Serve Deployment Statuses")
async def get_ray_serve_deployment_details():
    """
    Provides detailed status for each active Ray Serve deployment.
    """
    if not ray.is_initialized():
        logger.warning("Ray is not initialized. Cannot fetch deployment details.")
        raise HTTPException(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail="Ray is not initialized.")
    
    try:
        # Check if Serve is running/available
        # serve.status() might raise if Serve is not running, or client cannot connect
        # A more robust check might be needed depending on Ray version specifics
        try:
            serve_client = serve.api._get_global_client(raise_if_no_controller_running=True)
        except Exception as e:
            logger.warning(f"Ray Serve controller not running or client not available: {e}")
            raise HTTPException(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail="Ray Serve is not running or client not available.")

        deployment_statuses_proto = serve.get_deployment_statuses()
        detailed_statuses = []
        for status_proto in deployment_statuses_proto:
            replica_states_summary = {}
            # Accessing replica_states might differ slightly based on Ray version
            # Using hasattr for safer access
            proto_replica_states = None
            if hasattr(status_proto, 'replica_states') and isinstance(status_proto.replica_states, dict): # Older Ray versions (protobuf map)
                proto_replica_states = status_proto.replica_states
            elif hasattr(status_proto, 'deployment_status') and hasattr(status_proto.deployment_status, 'replica_states'): # Newer Ray versions
                 proto_replica_states = status_proto.deployment_status.replica_states
            
            if proto_replica_states:
                for state_val, count in proto_replica_states.items():
                    try:
                        # state_val might be an int or a string depending on Ray version
                        state_name = ReplicaState(int(state_val)).name if isinstance(state_val, (int, str)) and str(state_val).isdigit() else str(state_val)
                        replica_states_summary[state_name] = count
                    except ValueError: # Handle cases where state_val is not a valid ReplicaState enum member
                        logger.warning(f"Unknown replica state value '{state_val}' for deployment {status_proto.name}")
                        replica_states_summary[str(state_val)] = count


            deployment_config_dict = None
            if hasattr(status_proto, 'deployment_config') and status_proto.deployment_config and hasattr(status_proto.deployment_config, 'deployment_config_json') and status_proto.deployment_config.deployment_config_json:
                try:
                    deployment_config_dict = json.loads(status_proto.deployment_config.deployment_config_json)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse deployment_config_json for {status_proto.name}")
            
            app_name = None
            if hasattr(status_proto, 'application_name'): # Newer Ray versions
                app_name = status_proto.application_name
            elif deployment_config_dict and 'application_name' in deployment_config_dict: # Fallback for older versions if in config
                app_name = deployment_config_dict['application_name']


            detailed_statuses.append({
                "name": status_proto.name,
                "status": DeploymentStatus(status_proto.status).name if isinstance(status_proto.status, int) else str(status_proto.status),
                "message": status_proto.message,
                "replica_states": replica_states_summary,
                "deployment_config": deployment_config_dict,
                "application_name": app_name,
            })
        return detailed_statuses
    except ray.exceptions.RaySystemError as rse:
        logger.error(f"Ray system error when fetching deployment details: {rse}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Ray system error: {str(rse)}")
    except Exception as e:
        logger.error(f"Error fetching Ray Serve deployment details: {e}", exc_info=True)
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch Ray Serve deployment details: {str(e)}")

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
          logger.error(f"vLLM API test failed: Unexpected error: {e}")
          return ApiTestResponse(status="error", message="An unexpected error occurred during API test.", vllm_api_reachable=False, error_details=str(e))

@router.post("/models/{model_key}/unload", response_model=ServiceActionResponse, summary="Manually Unload a Model from Ray Serve")
async def unload_model_deployment(
   model_key: str = Path(..., description="The configuration key of the model whose deployment is to be unloaded.")
):
   """
   Attempts to manually unload (delete) a specific model's deployment from Ray Serve.
   This also sets the 'serve' status for the model in model_config.json to false.
   """
   logger.info(f"Request to manually unload model deployment for key: {model_key}")

   if not ray.is_initialized():
       logger.error("Ray is not initialized. Cannot unload model.")
       raise HTTPException(
           status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
           detail="Ray is not initialized. Cannot unload model."
       )
   try:
       serve_client = serve.api._get_global_client(raise_if_no_controller_running=True)
   except Exception as e:
       logger.error(f"Ray Serve controller not running or client not available: {e}")
       raise HTTPException(
           status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
           detail=f"Ray Serve is not running or client not available: {e}"
       )

   # Construct the deployment name. In ray_deployments.py, we use `/{key}`.
   # However, serve.delete might not need the leading slash if it's part of an application.
   # Let's try to get the actual deployment names from serve.
   
   actual_deployment_name = None
   try:
       all_deployments = serve.list_deployments() # Returns Dict[str, DeploymentInfo] in newer Ray
       # The keys in all_deployments are the full deployment names including app prefix if any.
       # Our model_key should match the deployment name suffix after the app name.
       # Example: if app is "default", deployment name might be "default_model_key" or just "model_key"
       # If app is "vllm_app", it might be "vllm_app_model_key".
       # The build_llm_deployments function creates deployments with names like "/{key}" when passed to serve.run
       # which often results in names like "default_key" or just "key" if no app name is specified in serve.run
       
       # Let's try common naming conventions
       possible_names = [model_key, f"/{model_key}"]
       # If an application name is used by serve.run (e.g. "vllm_app" in main.py, though currently not set)
       # we might need to check for "appname_modelkey"
       # For now, assume simple naming or direct match with model_key or /model_key

       for name_attempt in possible_names:
           if name_attempt in all_deployments:
               actual_deployment_name = name_attempt
               break
       
       # A more robust way might be to iterate all_deployments and check if model_key is a suffix
       if not actual_deployment_name:
           for dep_name in all_deployments.keys():
               if dep_name.endswith(model_key): # Check if model_key is a suffix
                   actual_deployment_name = dep_name
                   break
       
       if not actual_deployment_name:
           logger.warning(f"Deployment for model key '{model_key}' not found in Ray Serve. Current deployments: {list(all_deployments.keys())}")
           # Check if it's already marked as not served in config
           current_config = load_model_config()
           if model_key in current_config and isinstance(current_config[model_key], dict) and not current_config[model_key].get("serve", False):
                return ServiceActionResponse(
                   status="skipped",
                   message=f"Model '{model_key}' is already marked as not serving and no active deployment found."
               )
           return ServiceActionResponse(
               status="skipped",
               message=f"Deployment for model key '{model_key}' not found. It might be already unloaded or was never deployed with this key."
           )

       logger.info(f"Attempting to delete Ray Serve deployment: {actual_deployment_name}")
       serve.delete(actual_deployment_name, _blocking=True) # _blocking for synchronous delete
       logger.info(f"Successfully deleted Ray Serve deployment: {actual_deployment_name}")
       
       # Update the model's 'serve' status in config to false
       current_config = load_model_config()
       if model_key in current_config and isinstance(current_config[model_key], dict):
           current_config[model_key]['serve'] = False
           save_model_config(current_config)
           logger.info(f"Updated model_config.json: set 'serve' to false for '{model_key}' after unload.")
       else:
           logger.warning(f"Model key '{model_key}' not found in config or invalid format during unload cleanup, but Ray Serve deployment was deleted.")

       return ServiceActionResponse(status="ok", message=f"Model deployment '{actual_deployment_name}' successfully unloaded from Ray Serve and config updated to 'serve: false'.")

   except ray.exceptions.RayServeException as rse:
       logger.error(f"Ray Serve exception while trying to unload deployment for '{model_key}': {rse}", exc_info=True)
       # Check if it's because the deployment doesn't exist
       if "Deployment" in str(rse) and "does not exist" in str(rse):
            # Update config anyway if it was marked as serve: true
           current_config = load_model_config()
           if model_key in current_config and isinstance(current_config[model_key], dict) and current_config[model_key].get("serve", True):
               current_config[model_key]['serve'] = False
               save_model_config(current_config)
               logger.info(f"Deployment for '{model_key}' did not exist in Ray Serve. Marked as 'serve: false' in config.")
               return ServiceActionResponse(status="ok", message=f"Deployment for '{model_key}' not found in Ray Serve. Marked as not serving in config.")
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail=f"Ray Serve error unloading model deployment '{model_key}': {str(rse)}"
       )
   except Exception as e:
       logger.error(f"Error unloading Ray Serve deployment for '{model_key}': {e}", exc_info=True)
       raise HTTPException(
           status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail=f"Failed to unload model deployment '{model_key}': {str(e)}"
       )