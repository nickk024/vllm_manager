import logging
import logging.handlers # For file logging
import os
import sys # For stderr
from fastapi import FastAPI

# Import config to get VLLM_HOME early, LOGS_DIR is now dynamic
from .config import VLLM_HOME

# Import routers from the routers package
from .routers import models_router, download_router, service_router, monitoring_router

# Import NVML utils to ensure initialization and shutdown hook registration happens
from .utils import gpu_utils # This will trigger the NVML init check in gpu_utils

# --- Logging Configuration ---
# Get log directory from environment variable set by start.sh, fallback to default
LOG_DIR_DEFAULT = os.path.join(VLLM_HOME, "logs") # Default inside install dir
LOG_DIR = os.environ.get("VLLM_LOG_DIR", LOG_DIR_DEFAULT)
# Unified log file name
UNIFIED_LOG_FILE = os.path.join(LOG_DIR, "vllm_manager.log")
LOG_LEVEL = logging.DEBUG # Set log level to DEBUG for file

# Ensure log directory exists
log_dir_valid = False
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_dir_valid = True
    print(f"[Backend INFO] Logging directory set to: {LOG_DIR}")
except OSError as e:
     print(f"[Backend ERROR] Could not create or access log directory: {LOG_DIR}. Error: {e}", file=sys.stderr)
     LOG_DIR = None # Indicate failure


# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [Backend] %(message)s') # Added [Backend] prefix

# Get the root logger and configure handlers
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL) # Set root logger level to lowest level (DEBUG)

# Clear existing handlers (important if uvicorn adds its own)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create console handler
console_handler = logging.StreamHandler(sys.stderr) # Log to stderr
console_handler.setLevel(logging.INFO) # Keep console INFO level
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler) # Add console handler

# Create rotating file handler (to unified file) only if LOG_DIR is valid
if LOG_DIR:
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            UNIFIED_LOG_FILE, maxBytes=10*1024*1024, backupCount=5 # Shared file
        )
        file_handler.setLevel(LOG_LEVEL) # Set file handler level
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler) # Add file handler
        logging.info(f"--- Backend Logging configured (Console: INFO, File: {LOG_LEVEL} at {UNIFIED_LOG_FILE}) ---")
    except Exception as e:
         print(f"[Backend ERROR] Failed to create file log handler for {UNIFIED_LOG_FILE}. Error: {e}", file=sys.stderr)
         logging.error(f"Failed to create file log handler for {UNIFIED_LOG_FILE}. Error: {e}") # Log error to console
else:
     logging.warning("--- Backend Logging configured (Console only) ---")


# Get logger for this specific module (will inherit handlers from root)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---


# Create FastAPI app instance
app = FastAPI(
    title="vLLM Management Backend",
    description="API for managing vLLM models, service, and monitoring.",
    version="0.1.0"
)

# Include routers
logger.info("Including API routers...")
app.include_router(models_router.router, prefix="/api/v1")
app.include_router(download_router.router, prefix="/api/v1")
app.include_router(service_router.router, prefix="/api/v1")
app.include_router(monitoring_router.router, prefix="/api/v1")
logger.info("Routers included successfully.")

# Root endpoint for basic check
@app.get("/", tags=["Root"])
async def read_root():
    logger.debug("Root endpoint '/' accessed.")
    return {"message": "Welcome to the vLLM Management Backend API"}

# --- NVML Shutdown Hook ---
# The import of gpu_utils should handle registration if NVML is available.

logger.info("FastAPI application initialized.")

# Example: Run with uvicorn vllm_installer.backend.main:app --reload --port 8080