import logging
import logging.handlers # For file logging
import os
from fastapi import FastAPI

# Import config to get VLLM_HOME and LOGS_DIR early
from .config import VLLM_HOME, LOGS_DIR

# Import routers from the routers package
from .routers import models_router, download_router, service_router, monitoring_router

# Import NVML utils to ensure initialization and shutdown hook registration happens
from .utils import gpu_utils # This will trigger the NVML init check in gpu_utils

# --- Logging Configuration ---
LOG_FILE = os.path.join(LOGS_DIR, "vllm_backend.log")
LOG_LEVEL = logging.DEBUG # Set log level to DEBUG for file

# Ensure log directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Keep console INFO level
console_handler.setFormatter(formatter)

# Create rotating file handler
# Rotates log file when it reaches 10MB, keeps 5 backup files
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setLevel(LOG_LEVEL) # Set file handler level
file_handler.setFormatter(formatter)

# Get the root logger and add handlers
# Configure root logger to capture logs from all modules (FastAPI, uvicorn, our code)
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL) # Set root logger level to lowest level (DEBUG)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Get logger for this specific module
logger = logging.getLogger(__name__)
logger.info(f"--- Logging configured (Console: INFO, File: {LOG_LEVEL} at {LOG_FILE}) ---")
# --- End Logging Configuration ---


# Create FastAPI app instance
app = FastAPI(
    title="vLLM Management Backend",
    description="API for managing vLLM models, service, and monitoring.",
    version="0.1.0"
)

# Include routers
logger.info("Including API routers...")
app.include_router(models_router.router, prefix="/api/v1") # Removed tags here, defined in routers
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
# If a more explicit shutdown is needed within FastAPI's lifecycle:
# @app.on_event("shutdown")
# def shutdown_event():
#     if gpu_utils.NVML_AVAILABLE:
#         try:
#             gpu_utils.pynvml.nvmlShutdown()
#             logger.info("PyNVML shutdown successfully.")
#         except Exception as e:
#             logger.error(f"Error during PyNVML shutdown: {e}")

logger.info("FastAPI application initialized.")

# Example: Run with uvicorn vllm_installer.backend.main:app --reload --port 8080
# Note: Use the module path format for uvicorn