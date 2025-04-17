import os
from typing import Dict, Any, Optional
import logging

from ray import serve
# Import the vLLM engine integration for Ray Serve
# Ensure vllm is installed in the same environment Ray Serve runs in
try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    # Ray Serve's LLMServer simplifies things further if suitable
    # from ray.serve.llm import LLMServer, LLMApp
    # from ray.serve.openai_provider import OpenAIProvider
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


# Define the Ray Serve deployment for a single vLLM model instance
# This deployment will be managed (created, updated, deleted) by our FastAPI backend later.
# For now, it shows the basic structure.

# Option 1: Using lower-level vLLM AsyncEngine (more control)
# @serve.deployment(
#     # Configure resources: num_gpus=1 means this replica uses 1 GPU
#     # Adjust based on model requirements (TP size) and available hardware
#     # For TP > 1, you'd likely use multiple replicas coordinated by Ray? Or does vLLM handle it internally with num_gpus > 1? Needs research.
#     # Let's assume TP=1 for the initial always-on embedding model
#     num_gpus=1,
#     # Autoscaling config (optional)
#     # autoscaling_config={"min_replicas": 1, "max_replicas": 2},
# )
# class VLLMModelServer:
#     def __init__(self, model_key: str, model_config: Dict[str, Any]):
#         logger.info(f"Initializing VLLMModelServer for model key: {model_key}")
#         self.model_key = model_key
#         self.model_config = model_config
#         self.engine = None # Initialize later

#         model_path = os.path.join(MODELS_DIR, self.model_key)
#         if not os.path.isdir(model_path):
#              logger.error(f"Model directory not found at {model_path} for key {self.model_key}")
#              # Handle error - perhaps raise exception to prevent deployment?
#              raise ValueError(f"Model directory not found for {self.model_key}")

#         # Extract engine args from model_config or use defaults
#         engine_args = AsyncEngineArgs(
#             model=model_path, # Use local path
#             served_model_name=self.model_key, # Identify model in API calls
#             tensor_parallel_size=self.model_config.get("tensor_parallel_size", 1),
#             dtype=self.model_config.get("dtype", "auto"),
#             max_model_len=self.model_config.get("max_model_len"),
#             # Add other relevant args: gpu_memory_utilization, quantization, etc.
#             gpu_memory_utilization=0.90,
#             # Important for embeddings or non-chat models if needed:
#             # enforce_eager=True, # May be needed for some models/ops
#             # trust_remote_code=True # If model requires it
#         )
#         logger.info(f"Engine Args for {self.model_key}: {engine_args}")
#         self.engine = AsyncLLMEngine.from_engine_args(engine_args)
#         logger.info(f"vLLM engine initialized for {self.model_key}.")

#     # Define methods to handle requests, e.g., compatible with OpenAI API
#     # This requires building the logic to map incoming requests (like /v1/chat/completions)
#     # to the self.engine.generate() method and formatting the response.
#     # Ray Serve's LLMApp/OpenAIProvider might simplify this significantly.
#     async def generate(self, request_data: Dict[str, Any]):
#         # Placeholder: Implement actual call to self.engine.generate or other methods
#         logger.info(f"Received generate request for {self.model_key}: {request_data}")
#         # Example: await self.engine.generate(...)
#         return {"model": self.model_key, "text": "Generated text placeholder"}

#     # Health check (optional but recommended)
#     async def check_health(self):
#         # Basic check: engine exists
#         if self.engine is None:
#              raise RuntimeError("vLLM Engine not initialized")
#         # More advanced checks possible here
#         logger.debug(f"Health check passed for {self.model_key}")


# Option 2: Using Ray Serve's higher-level LLMApp (Simpler for OpenAI compatibility)
# This requires defining the models within the LLMApp configuration.
# Dynamic add/remove might be handled by updating the app config via serve.run(app)

# Example structure (needs refinement based on how we trigger loading)
# from ray.serve.llm import LLMApp
# from ray.serve.openai_provider import OpenAIProvider

# def build_llm_app(models_to_serve: Dict[str, Dict[str, Any]]) -> LLMApp:
#     """Builds the LLMApp based on currently configured models to serve."""
#     app_config = {}
#     for config_key, model_conf in models_to_serve.items():
#         model_path = os.path.join(MODELS_DIR, config_key)
#         if not os.path.isdir(model_path):
#             logger.warning(f"Skipping model {config_key}: Directory not found at {model_path}")
#             continue

#         # Map our config to LLMApp's expected format
#         app_config[config_key] = {
#             "model": model_path,
#             "engine_config": {
#                 "tensor_parallel_size": model_conf.get("tensor_parallel_size", 1),
#                 "max_model_len": model_conf.get("max_model_len"),
#                 "dtype": model_conf.get("dtype", "auto"),
#                 "gpu_memory_utilization": 0.90, # Make configurable?
#                 # Add other engine args here
#             },
#             # Add provider info if needed (defaults to OpenAIProvider)
#             # "provider": OpenAIProvider(),
#             # Add route prefix if needed
#             # "route_prefix": f"/model/{config_key}"
#         }
#         logger.info(f"Prepared app config for model: {config_key}")

#     if not app_config:
#          logger.error("No valid models found to serve in LLMApp config.")
#          # Return a dummy app or raise error?
#          return LLMApp({}) # Return empty app

#     # LLMApp takes **kwargs for models
#     return LLMApp(**app_config)


# --- Ray Serve LLMApp Builder for Dynamic Model Serving ---

from typing import Dict
import os
import logging

try:
    from ray.serve.llm import LLMApp
except ImportError:
    LLMApp = None

# Removed BASE_MODEL_KEY constant

def build_llm_app(full_config: Dict[str, Dict]) -> Optional["LLMApp"]:
   """
   Build a Ray Serve LLMApp for OpenAI-compatible serving based on the provided configuration.

   Args:
       full_config: The complete model configuration dictionary loaded from model_config.json.

   Returns:
       An LLMApp instance configured with models marked "serve: true" and downloaded,
       or None if no models are ready to be served or LLMApp is unavailable.
   """
   if LLMApp is None:
       logger.error("Cannot build LLMApp: ray[serve] LLMApp is not available. Please install ray[serve].")
       return None

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
       logger.warning("No models configured, marked to serve, AND downloaded found. Returning empty LLMApp config (Ray Serve might not start any models).")
       # Returning an empty app might be okay, Ray Serve handles it.
       # Alternatively, could return None and handle upstream in main.py
       # Let's return None to be explicit that no app should be deployed.
       return None
       # raise RuntimeError("No valid models found to serve in LLMApp config.") # Or raise error?

   logger.info(f"Prepared LLMApp config with {models_prepared_count} model(s).")

   # LLMApp takes **kwargs for models
   return LLMApp(**app_config)

# --- Idle Timeout/Unloading Placeholder ---
# The actual idle timeout and unloading logic will be managed by the backend,
# which will update the LLMApp deployment via serve.run() as needed.
# This file provides the builder function for the current set of models to serve.

# --- Placeholder for Deployment Logic ---
# This file defines *how* to deploy, but the actual deployment
# will likely be triggered from backend/main.py or via `serve deploy` CLI.

# Example of how deployment might be triggered (e.g., in main.py):
#
# import ray
# from ray import serve
# from backend.ray_deployments import build_llm_app # Or VLLMModelServer
# from backend.config import load_model_config, MODELS_DIR
#
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Connecting to Ray cluster...")
#     # Connect to the running Ray cluster started by start.sh
#     # address="auto" assumes head node is discoverable
#     ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
#     logger.info("Connected to Ray.")
#
#     # Load config and find models marked with "serve": true
#     models_to_serve = {}
#     full_config = load_model_config()
#     for key, conf in full_config.items():
#         if conf.get("serve", False): # Check the 'serve' flag
#             model_path = os.path.join(MODELS_DIR, key)
#             if os.path.isdir(model_path): # Check if downloaded
#                 models_to_serve[key] = conf
#             else:
#                 logger.warning(f"Model '{key}' marked to serve but not downloaded. Skipping.")
#
#     if not models_to_serve:
#         logger.warning("No models configured or downloaded to be served initially.")
#         # Decide: deploy empty app? Or wait for API call?
#         # For now, let's assume we need at least one model (like nomic) configured & served
#         # Example: Force load nomic if configured?
#         nomic_key = "nomic_embed_text_v1_5" # Example key
#         if nomic_key in full_config and os.path.isdir(os.path.join(MODELS_DIR, nomic_key)):
#              models_to_serve[nomic_key] = full_config[nomic_key]
#              logger.info(f"Ensuring '{nomic_key}' is served.")
#         else:
#              logger.error("No models ready to serve, including the default embedding model. Ray Serve deployment might fail or be empty.")
#              # Handle this case - maybe don't deploy?
#
#     # Build and deploy the application using LLMApp (Option 2)
#     # This replaces the previous systemd service start
#     if models_to_serve:
#          llm_application = build_llm_app(models_to_serve)
#          # Deploy the application. `blocking=False` allows FastAPI to continue starting.
#          # The route_prefix determines where the OpenAI API is exposed.
#          serve.run(llm_application, name="vllm", route_prefix="/", host="0.0.0.0", port=8000, blocking=False)
#          logger.info("Ray Serve deployment initiated for vLLM models on port 8000.")
#     else:
#          logger.warning("Skipping initial Ray Serve deployment as no models are ready.")

# Note: The actual deployment trigger needs to be integrated into main.py's startup logic.
# This file just defines the potential deployment structures.