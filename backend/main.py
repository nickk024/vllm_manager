import logging
import logging.handlers
import os
import sys
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Import config early for paths
from .config import VLLM_HOME, LOGS_DIR, MODELS_DIR, load_model_config

# Import routers
from .routers import models_router, download_router, service_router, monitoring_router

# --- Ray and Deployment Imports ---
import ray
from ray import serve
try:
    # Use the LLMApp approach for simplicity
    from ray.serve.llm import LLMApp
    # We might not need OpenAIProvider explicitly if LLMApp handles it
    # from ray.serve.openai_provider import OpenAIProvider
    RAY_SERVE_LLM_AVAILABLE = True
except ImportError:
    logging.error("Ray Serve LLM components (LLMApp) not found. pip install 'ray[serve]'? vLLM deployment will fail.")
    RAY_SERVE_LLM_AVAILABLE = False
    # Define dummy class to prevent import errors later if needed
    class LLMApp: pass
# --- End Ray Imports ---


# --- Logging Configuration ---
LOG_DIR_DEFAULT = os.path.join(VLLM_HOME, "logs")
LOG_DIR = os.environ.get("VLLM_LOG_DIR", LOG_DIR_DEFAULT)
UNIFIED_LOG_FILE = os.path.join(LOG_DIR, "vllm_manager.log")
LOG_LEVEL = logging.DEBUG
log_dir_valid = False
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_dir_valid = True
    print(f"[Backend INFO] Logging directory set to: {LOG_DIR}")
except OSError as e:
     print(f"[Backend ERROR] Could not create/access log directory: {LOG_DIR}. Error: {e}", file=sys.stderr)
     LOG_DIR = None
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [Backend] %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
if LOG_DIR:
    try:
        file_handler = logging.handlers.RotatingFileHandler(UNIFIED_LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"--- Backend Logging configured (Console: INFO, File: {LOG_LEVEL} at {UNIFIED_LOG_FILE}) ---")
    except Exception as e:
         print(f"[Backend ERROR] Failed to create file log handler for {UNIFIED_LOG_FILE}. Error: {e}", file=sys.stderr)
         logging.error(f"Failed to create file log handler for {UNIFIED_LOG_FILE}. Error: {e}")
else:
     logging.warning("--- Backend Logging configured (Console only) ---")
logger = logging.getLogger(__name__) # Logger for this module
serve_logger = logging.getLogger("ray.serve") # Get Ray Serve's logger
serve_logger.setLevel(LOG_LEVEL) # Ensure serve logs are captured at desired level
if log_dir_valid: serve_logger.addHandler(file_handler) # Add file handler to serve logger too
if not any(isinstance(h, logging.StreamHandler) for h in serve_logger.handlers):
     serve_logger.addHandler(console_handler) # Ensure serve logs also go to console if needed
# --- End Logging Configuration ---


# --- Ray Serve Application Definition ---
# Use the builder from ray_deployments.py for LLMApp construction
from .ray_deployments import build_llm_app

# --- FastAPI App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    logger.info("FastAPI application startup...")
    logger.info("Connecting to Ray cluster (or starting one if needed)...")
    try:
        # Connect to the running Ray cluster started by start.sh
        # address="auto" assumes head node is discoverable
        # namespace="serve" is important for Ray Serve components to find each other
        ray.init(address="auto", namespace="serve", ignore_reinit_error=True, logging_level=logging.WARNING) # Reduce Ray's verbosity
        logger.info("Connected to Ray.")

        # Build the initial LLMApp configuration using the shared builder
        full_config = load_model_config()
        llm_application = build_llm_app(full_config)

        if llm_application:
            # Deploy the application using serve.run()
            # This replaces the previous systemd service start for vLLM itself
            # The host/port here are for Ray Serve's API endpoint (proxy)
            # We want this to be the main inference endpoint (e.g., 8000)
            serve_host = "0.0.0.0"
            serve_port = 8000 # Standard OpenAI API port
            logger.info(f"Deploying Ray Serve application 'vllm_app' on {serve_host}:{serve_port}...")
            try:
                 # Define deployment options here
                 # Assign GPUs based on max TP size needed? Or let Ray schedule?
                 # For simplicity, let Ray schedule based on total GPUs available initially.
                 # TODO: Refine resource allocation based on max_required_tp
                 serve.run(
                      llm_application,
                      name="vllm_app", # Name for the Ray Serve application
                      route_prefix="/", # Expose at root (e.g., /v1/models)
                      host=serve_host,
                      port=serve_port,
                      blocking=False # Allow FastAPI startup to continue
                 )
                 logger.info("Ray Serve deployment initiated.")
            except Exception as e:
                 logger.error(f"Failed to deploy Ray Serve application: {e}", exc_info=True)
                 # Should we exit FastAPI startup?
        else:
             logger.warning("Could not build LLMApp, skipping Ray Serve deployment.")

    except Exception as e:
        logger.error(f"Error during Ray initialization or initial deployment: {e}", exc_info=True)
        # Continue starting FastAPI management API even if Ray deployment fails? Or exit?
        # Let's continue for now, management API might still be useful.

    yield # App runs here

    # Runs on shutdown
    logger.info("FastAPI application shutdown...")
    try:
        logger.info("Shutting down Ray Serve...")
        serve.shutdown()
        logger.info("Ray Serve shutdown complete.")
        # Ray cluster itself might still be running if started by start.sh separately
        # ray.shutdown() # Optional: shutdown the Ray connection/cluster?
    except Exception as e:
        logger.error(f"Error during Ray Serve shutdown: {e}", exc_info=True)


# Create FastAPI app instance with lifespan manager
app = FastAPI(
    title="vLLM Management Backend (Ray Serve)",
    description="API for managing vLLM models (via Ray Serve), downloads, and monitoring.",
    version="0.2.0-ray",
    lifespan=lifespan # Use the lifespan context manager
)

# Include management routers (adjust prefixes/tags if needed)
logger.info("Including Management API routers...")
app.include_router(models_router.router, prefix="/api/v1/manage") # Prefix management routes
app.include_router(download_router.router, prefix="/api/v1/manage")
app.include_router(service_router.router, prefix="/api/v1/manage") # Service router now controls Ray Serve app? Needs update.
app.include_router(monitoring_router.router, prefix="/api/v1/manage")
logger.info("Management routers included successfully.")

# Root endpoint for basic check
@app.get("/", tags=["Root"])
async def read_root():
    logger.debug("Root endpoint '/' accessed.")
    return {"message": "Welcome to the vLLM Management Backend API (Ray Serve Edition)"}


logger.info("FastAPI application definition complete.")

# Example: Run with uvicorn backend.main:app --reload --port 8080