import logging
import logging.handlers
import os
import sys
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Import config early for paths
from .config import VLLM_HOME, LOGS_DIR, MODELS_DIR, load_model_config

# --- Custom Unified Logger Configuration ---
import pathlib

# Define log path relative to the project root, using the unified name
# Use os.path.abspath to ensure the path is correctly resolved
# Ensure absolute path for logging
UNIFIED_LOG_PATH = os.path.abspath(os.path.join(LOGS_DIR, "vllm_manager_app.log"))
BACKEND_LOG_DIR = os.path.dirname(UNIFIED_LOG_PATH)
LOG_LEVEL = logging.DEBUG # Keep DEBUG level for file logging

# Ensure the project log directory exists
# Create logger instance *before* trying to use it
logger = logging.getLogger(__name__) # Get logger for this module first

try:
    # Ensure the log directory exists with proper permissions
    pathlib.Path(BACKEND_LOG_DIR).mkdir(parents=True, exist_ok=True)
    os.chmod(BACKEND_LOG_DIR, 0o755)  # Ensure proper permissions
    # Use the already defined logger instance
    logger.info(f"Log directory ensured: {BACKEND_LOG_DIR}")
except Exception as e:
    # Use print for critical failure before logging is fully set up
    print(f"[CRITICAL] Could not create/access log directory: {BACKEND_LOG_DIR}. Error: {e}", file=sys.stderr)
    sys.exit(1)  # Fail fast if logging setup fails

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [Backend] %(message)s')

# Set up the root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler) # Clear existing handlers
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO) # Console logs INFO and above
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

try:
    # Use the unified path for the file handler
    file_handler = logging.handlers.RotatingFileHandler(UNIFIED_LOG_PATH, maxBytes=20*1024*1024, backupCount=10)
    file_handler.setLevel(LOG_LEVEL) # Log DEBUG and above to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logger.info(f"--- Backend Logging configured (Console: INFO, File: {LOG_LEVEL} at {UNIFIED_LOG_PATH}) ---")
except Exception as e:
    logger.error(f"Failed to create file log handler for {UNIFIED_LOG_PATH}. Error: {e}", exc_info=True)
    # Fallback to console if file logging fails, but don't exit

# Global exception hook to log uncaught exceptions
def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    # Use the root logger for uncaught exceptions
    logging.getLogger().critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_uncaught_exceptions
# --- End Custom Unified Logger Configuration ---

# Import routers AFTER logger is configured
from .routers import models_router, download_router, service_router, monitoring_router

# --- Ray and Deployment Imports ---
# Now safe to import Ray components, logger is available for error handling
import ray
from ray import serve
try:
    # Basic Ray/Serve import check
    import pyarrow
    # No need to import LLMApp here anymore
    logger.info("Ray and Ray Serve core components seem available.")

    # Validate Ray connection (moved inside lifespan)

    # Attach the special file handler to Ray Serve logger if Ray Serve is available
    # Do this *after* Ray is initialized in lifespan to ensure logger exists
    # Moved logger setup to lifespan

except ImportError as e:
    logger.error(f"Core Ray or Serve component missing: {e}. LLM deployment will likely fail.", exc_info=True)
    # Allow FastAPI to start, but deployment will fail later in lifespan
# --- End Ray Imports ---


# --- Ray Serve Application Definition ---
# Use the builder from ray_deployments.py for LLM deployment construction
from .ray_deployments import build_llm_deployments # Import the renamed function

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

        # Configure Ray Serve logger *after* Ray init and *before* deployment
        try:
            serve_logger = logging.getLogger("ray.serve")
            serve_logger.setLevel(LOG_LEVEL) # Ensure Ray Serve logs at the desired file level
            # Add the unified file handler to Ray Serve logger
            if 'file_handler' in locals() and file_handler not in serve_logger.handlers:
                 serve_logger.addHandler(file_handler)
            # Keep console handler for Ray Serve as well, if not already present
            if not any(isinstance(h, logging.StreamHandler) for h in serve_logger.handlers):
                serve_logger.addHandler(console_handler) # Use the same console handler
            logger.info("Ray Serve logger configured.")
        except NameError:
             logger.warning("File handler not defined, skipping adding it to Ray Serve logger.")
        except Exception as e:
            logger.error(f"Error configuring Ray Serve logger: {e}", exc_info=True)


        # Build the dictionary of LLM deployments using the shared builder
        full_config = load_model_config()
        llm_deployments = build_llm_deployments(full_config)

        if llm_deployments:
            # Deploy the dictionary of deployments using serve.run()
            # Keys in the dict become route prefixes (e.g., /model_key)
            serve_host = "0.0.0.0"
            serve_port = 8000 # Standard OpenAI API port
            logger.info(f"Deploying {len(llm_deployments)} Ray Serve LLM deployment(s) on {serve_host}:{serve_port}...")
            try:
                 # serve.run accepts a dictionary of {route_prefix: deployment}
                 serve.run(
                      llm_deployments,
                      # name="vllm_app", # Name is less relevant when deploying dict
                      # route_prefix="/", # Prefixes are defined by dict keys
                      host=serve_host,
                      port=serve_port,
                      blocking=False # Allow FastAPI startup to continue
                 )
                 logger.info("Ray Serve deployment initiation request sent.")
            except Exception as e:
                 logger.error(f"Failed to deploy Ray Serve LLM deployments: {e}", exc_info=True)
                 # Should we exit FastAPI startup?
        else:
             logger.warning("No valid LLM deployments built, skipping Ray Serve deployment.")

    except ImportError as e:
         logger.error(f"ImportError during Ray initialization or deployment build: {e}. Cannot deploy models.", exc_info=True)
    except Exception as e:
        logger.error(f"Error during Ray initialization or initial deployment setup: {e}", exc_info=True)
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