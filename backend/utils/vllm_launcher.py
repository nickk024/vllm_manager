#!/usr/bin/env python3
"""
vLLM Launcher Script for Systemd

Reads the active model configuration and launches the vLLM OpenAI API server
with the appropriate tensor_parallel_size and other model-specific parameters.
"""

import os
import sys
import json
import logging
import argparse
import subprocess

# Configure logging for the launcher script itself
# Log to stdout/stderr for systemd journal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Launcher] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vllm_launcher")

# --- Import config AFTER setting up logger ---
# Use relative import assuming this runs within the backend package context
# If run standalone, PYTHONPATH might need adjustment or use absolute paths
try:
    from ..config import (
        VLLM_HOME, MODELS_DIR, CONFIG_PATH,
        load_model_config, read_active_model_key
    )
except ImportError:
    # Fallback for potential standalone execution or path issues
    logger.error("Failed relative import. Trying absolute paths based on script location.")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_DIR = os.path.dirname(SCRIPT_DIR) # Assumes utils is one level down
    PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
    sys.path.insert(0, PROJECT_ROOT) # Add project root to path
    try:
         from backend.config import (
             VLLM_HOME, MODELS_DIR, CONFIG_PATH,
             load_model_config, read_active_model_key
         )
    except ImportError as e:
         logger.critical(f"Cannot import config module even with path adjustment: {e}")
         sys.exit("Launcher Error: Cannot import configuration.")


# --- Configuration ---
# Paths are now imported from config.py
VENV_PYTHON = os.path.join(VLLM_HOME, "venv", "bin", "python") # Path to venv python

# Default vLLM server settings (can be overridden by args or model config)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_GPU_MEM_UTIL = 0.90 # Default higher for vLLM


def main():
    parser = argparse.ArgumentParser(description="vLLM Dynamic Launcher")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host for vLLM server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for vLLM server")
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL, help="GPU memory utilization")
    # Add other potential vLLM args if needed

    args = parser.parse_args()

    logger.info("--- vLLM Launcher Starting ---")

    # 1. Get the key of the model that should be active using config helper
    active_model_key = read_active_model_key() # Uses centralized function
    if not active_model_key:
        logger.error("No active model key found. Cannot determine which model to launch.")
        sys.exit("Launcher Error: Active model not specified.")

    # 2. Load the main model configuration using config helper
    model_config_data = load_model_config() # Uses centralized function
    if not model_config_data:
        sys.exit(f"Launcher Error: Failed to load model config from {CONFIG_PATH}")

    # 3. Find the config for the active model
    active_model_config = model_config_data.get(active_model_key)
    if not active_model_config:
        logger.error(f"Active model key '{active_model_key}' not found in configuration file {CONFIG_PATH}.")
        sys.exit(f"Launcher Error: Configuration for active model '{active_model_key}' not found.")

    # 4. Extract necessary parameters
    model_id = active_model_config.get("model_id")
    if not model_id:
         sys.exit(f"Launcher Error: 'model_id' missing for active model '{active_model_key}'.")

    tensor_parallel_size = active_model_config.get("tensor_parallel_size", 1)
    max_model_len = active_model_config.get("max_model_len")
    dtype = active_model_config.get("dtype")

    # Construct the full path to the model directory
    model_path = os.path.join(MODELS_DIR, active_model_key)
    if not os.path.isdir(model_path):
         logger.error(f"Model directory for active model '{active_model_key}' not found at {model_path}. Has it been downloaded?")
         logger.warning(f"Attempting to launch using model_id '{model_id}' directly.")
         model_arg = model_id # Use HF ID
    else:
         model_arg = model_path # Use local path

    logger.info(f"Preparing to launch model: {active_model_key} ({model_id})")
    logger.info(f"  Model Path/ID: {model_arg}")
    logger.info(f"  Tensor Parallel Size: {tensor_parallel_size}")
    if max_model_len: logger.info(f"  Max Model Length: {max_model_len}")
    if dtype: logger.info(f"  DType: {dtype}")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  GPU Memory Utilization: {args.gpu_memory_utilization}")

    # 5. Construct the vLLM command
    vllm_command = [
        VENV_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--host", args.host,
        "--port", str(args.port),
        "--model", model_arg,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    if max_model_len:
        vllm_command.extend(["--max-model-len", str(max_model_len)])
    if dtype:
        vllm_command.extend(["--dtype", dtype])

    # Add other necessary vLLM flags here

    logger.info(f"Executing command: {' '.join(vllm_command)}")

    # 6. Execute the vLLM server command using execvp
    try:
        # Ensure environment variables like HF_HOME are set by systemd if needed
        # (The template includes HF_HOME)
        os.execvp(vllm_command[0], vllm_command)
        # If execvp returns, it means an error occurred
        logger.error("Failed to execute vLLM server via execvp.")
        sys.exit("Launcher Error: Failed to execvp vLLM process.")

    except FileNotFoundError:
         logger.error(f"Error: vLLM python executable not found at {VENV_PYTHON}. Is the virtual environment correctly set up?")
         sys.exit("Launcher Error: Python executable not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred trying to launch vLLM: {e}")
        sys.exit(f"Launcher Error: {e}")

if __name__ == "__main__":
    main()