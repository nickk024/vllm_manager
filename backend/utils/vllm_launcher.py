#!/usr/bin/env python3
"""
vLLM Launcher Script for Systemd (Multi-Model Serving via Ray Serve)

Reads the model configuration file, identifies models marked to be served
and downloaded, calculates the maximum required tensor parallelism, and
launches the vLLM OpenAI API server with all specified models loaded.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from typing import List, Dict, Any

# All logs from this module are unified via the root logger in backend/main.py.
# Do not set up per-module file handlers or basicConfig here.
import logging
logger = logging.getLogger()

# --- Import config AFTER setting up logger ---
try:
    from backend.config import ( # Use absolute import assuming PYTHONPATH is set
        VLLM_HOME, MODELS_DIR, CONFIG_PATH,
        load_model_config
    )
    from backend.utils.gpu_utils import get_gpu_count # Import GPU count utility
except ImportError:
    logger.error("Failed backend import. Trying relative paths based on script location.")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    try:
         from backend.config import (
             VLLM_HOME, MODELS_DIR, CONFIG_PATH,
             load_model_config
         )
         from backend.utils.gpu_utils import get_gpu_count
    except ImportError as e:
         logger.critical(f"Cannot import config/utils module even with path adjustment: {e}")
         sys.exit("Launcher Error: Cannot import configuration/utils.")


# --- Configuration ---
VENV_PYTHON = os.path.join(VLLM_HOME, "venv", "bin", "python")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000 # Default vLLM OpenAI API port (Ray Serve will proxy to this if used)
DEFAULT_GPU_MEM_UTIL = 0.90


def main():
    parser = argparse.ArgumentParser(description="vLLM Multi-Model Launcher")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host for vLLM server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for vLLM server")
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL, help="GPU memory utilization")
    # Add --served-model-name? vLLM uses dir name by default if multiple --model args used.
    # parser.add_argument("--served-model-name", action="append", help="Name(s) for served models")

    args = parser.parse_args()

    logger.info("--- vLLM Launcher Starting (Multi-Model Mode) ---")

    # 1. Load the main model configuration
    model_config_data = load_model_config()
    if not model_config_data:
        logger.error(f"Model config file is empty or failed to load: {CONFIG_PATH}. Cannot start.")
        sys.exit(1) # Exit with error

    # 2. Filter for models to be served
    models_to_launch: Dict[str, Dict[str, Any]] = {}
    max_tp_size = 1 # Default TP size

    logger.info("Identifying models to launch based on config 'serve' flag and download status...")
    for key, conf in model_config_data.items():
        if not isinstance(conf, dict):
            logger.warning(f"Skipping invalid config entry for key '{key}': not a dictionary.")
            continue

        serve_flag = conf.get("serve", False)
        model_path = os.path.join(MODELS_DIR, key)
        is_downloaded = os.path.isdir(model_path) and bool(os.listdir(model_path))

        if serve_flag and is_downloaded:
            logger.info(f"Model '{key}' marked to serve and is downloaded.")
            models_to_launch[key] = conf
            # Update max TP size needed
            model_tp = conf.get("tensor_parallel_size", 1)
            if model_tp > max_tp_size:
                logger.info(f"Updating max TP size from {max_tp_size} to {model_tp} based on model '{key}'.")
                max_tp_size = model_tp
        elif serve_flag and not is_downloaded:
             logger.warning(f"Model '{key}' marked to serve but not downloaded. Skipping.")
        else:
             logger.debug(f"Model '{key}' not marked to serve or not downloaded. Skipping.")

    if not models_to_launch:
        logger.error("No models configured, marked to serve, AND downloaded found. Cannot start vLLM service.")
        sys.exit(0) # Exit cleanly, nothing to serve

    # 3. Verify max_tp_size against available GPUs
    available_gpus = get_gpu_count()
    if max_tp_size > available_gpus:
         logger.error(f"Required max tensor parallel size ({max_tp_size}) exceeds available GPUs ({available_gpus}). Cannot start.")
         # TODO: Optionally, could try to proceed with available_gpus, but might fail for large models.
         sys.exit(1)

    logger.info(f"Calculated maximum required Tensor Parallel Size: {max_tp_size}")
    logger.info(f"Models to launch: {list(models_to_launch.keys())}")

    # 4. Construct the vLLM command
    vllm_command = [
        VENV_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(max_tp_size), # Use the calculated max TP size globally
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    # Add multiple --model arguments
    for key in models_to_launch.keys():
        model_path = os.path.join(MODELS_DIR, key)
        vllm_command.extend(["--model", model_path])
        # Optionally add served-model-name if needed, but directory name is often default
        # vllm_command.extend(["--served-model-name", key])

    # Add other global args if needed (e.g., max_num_seqs)
    # Note: Per-model args like max_model_len, dtype are not directly settable here
    # for the OpenAI entrypoint when loading multiple models this way. vLLM
    # will likely use the config found within each model's directory.
    logger.warning("Note: Per-model settings like max_model_len/dtype are determined by config within model directory when loading multiple models.")

    logger.info(f"Executing command: {' '.join(vllm_command)}")

    # 5. Execute the vLLM server command using execvp
    try:
        os.execvp(vllm_command[0], vllm_command)
        logger.error("Failed to execute vLLM server via execvp.")
        sys.exit("Launcher Error: Failed to execvp vLLM process.")
    except FileNotFoundError:
         logger.error(f"Error: vLLM python executable not found at {VENV_PYTHON}.")
         sys.exit("Launcher Error: Python executable not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred trying to launch vLLM: {e}")
        sys.exit(f"Launcher Error: {e}")

if __name__ == "__main__":
    main()