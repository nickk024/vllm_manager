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


# --- Custom Unified Logger Configuration ---
import pathlib

# Define log path relative to the project root, using the unified name
# Use os.path.abspath to ensure the path is correctly resolved
UNIFIED_LOG_PATH = os.path.join(LOGS_DIR, "vllm_manager_app.log")  # Use centralized log dir
BACKEND_LOG_DIR = os.path.dirname(UNIFIED_LOG_PATH) # Keep variable name for clarity below, but uses unified path
LOG_LEVEL = logging.DEBUG # Keep DEBUG level for file logging

# Ensure the project log directory exists
try:
    pathlib.Path(BACKEND_LOG_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[Backend INFO] Log directory ensured: {BACKEND_LOG_DIR}")
except Exception as e:
    print(f"[Backend ERROR] Could not create/access log directory: {BACKEND_LOG_DIR}. Error: {e}", file=sys.stderr)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [Backend] %(message)s')

# Set up the root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

try:
    # Use the unified path for the file handler
    file_handler = logging.handlers.RotatingFileHandler(UNIFIED_LOG_PATH, maxBytes=20*1024*1024, backupCount=10)
    file_handler.setLevel(LOG_LEVEL) # Log DEBUG and above to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logging.info(f"--- Backend Logging configured (Console: INFO, File: {LOG_LEVEL} at {UNIFIED_LOG_PATH}) ---")
except Exception as e:
    print(f"[Backend ERROR] Failed to create file log handler for {UNIFIED_LOG_PATH}. Error: {e}", file=sys.stderr)
    logging.error(f"Failed to create file log handler for {UNIFIED_LOG_PATH}. Error: {e}")

# Attach the special file handler to Ray Serve logger as well
serve_logger = logging.getLogger("ray.serve")
serve_logger.setLevel(LOG_LEVEL) # Ensure Ray Serve logs at the desired file level
# Add the unified file handler to Ray Serve logger
serve_logger.addHandler(file_handler)
# Keep console handler for Ray Serve as well, if not already present
if not any(isinstance(h, logging.StreamHandler) for h in serve_logger.handlers):
    serve_logger.addHandler(console_handler) # Use the same console handler

# Optionally, attach to all FastAPI routers if needed
logger = logging.getLogger(__name__)

# Global exception hook to log uncaught exceptions
def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_uncaught_exceptions
# --- End Custom Unified Logger Configuration ---


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
        # Cluster connection with resource validation
        try:
            ray.init(address="auto", namespace="serve", ignore_reinit_error=True, logging_level=logging.WARNING)
        except ConnectionError:
            logger.warning("No existing Ray cluster found - starting local instance")
            ray.init(
                address="local",
                namespace="serve",
                num_cpus=max(1, os.cpu_count() // 2),  # Dynamic resource allocation
                num_gpus=1 if os.getenv('CUDA_VISIBLE_DEVICES') else 0,
                runtime_env={"working_dir": VLLM_HOME},
                ignore_reinit_error=True,
                logging_level=logging.WARNING
            )
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
app.include_router(service_router.router, prefix="/api/v1/manage/service")
app.include_router(monitoring_router.router, prefix="/api/v1/manage")
logger.info("Management routers included successfully.")

# Root endpoint for basic check
@app.get("/", tags=["Root"])
async def read_root():
    logger.debug("Root endpoint '/' accessed.")
    return {"message": "Welcome to the vLLM Management Backend API (Ray Serve Edition)"}


logger.info("FastAPI application definition complete.")

# Example: Run with uvicorn backend.main:app --reload --port 8080