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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Launcher] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout for systemd
)
logger = logging.getLogger("vllm_launcher")

# --- Configuration ---
# These should match the paths used by the backend and installer
VLLM_HOME_DEFAULT = "/opt/vllm"
VLLM_HOME = os.environ.get("VLLM_HOME", VLLM_HOME_DEFAULT)
CONFIG_DIR = os.path.join(VLLM_HOME, "config")
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "model_config.json")
ACTIVE_MODEL_FILE = os.path.join(CONFIG_DIR, "active_model.txt")
MODELS_DIR = os.path.join(VLLM_HOME, "models")
VENV_PYTHON = os.path.join(VLLM_HOME, "venv", "bin", "python") # Path to venv python

# Default vLLM server settings (can be overridden by args or model config)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_GPU_MEM_UTIL = 0.90 # Default higher for vLLM

def load_json_config(path: str) -> dict:
    """Loads JSON config from a file."""
    logger.info(f"Loading JSON config from: {path}")
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        return {}
    try:
        with open(path, 'r') as f:
            content = f.read()
            if not content.strip():
                logger.error(f"Config file is empty: {path}")
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        return {}

def get_active_model_key() -> str | None:
    """Reads the active model key from the tracking file."""
    logger.info(f"Reading active model key from: {ACTIVE_MODEL_FILE}")
    if not os.path.exists(ACTIVE_MODEL_FILE):
        logger.warning(f"Active model file not found: {ACTIVE_MODEL_FILE}")
        return None
    try:
        with open(ACTIVE_MODEL_FILE, 'r') as f:
            key = f.read().strip()
            if not key:
                 logger.warning(f"Active model file is empty: {ACTIVE_MODEL_FILE}")
                 return None
            logger.info(f"Found active model key: {key}")
            return key
    except Exception as e:
        logger.error(f"Error reading active model file {ACTIVE_MODEL_FILE}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="vLLM Dynamic Launcher")
    # Allow overriding defaults via command line if needed (e.g., from systemd)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host for vLLM server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for vLLM server")
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL, help="GPU memory utilization")
    # Add other potential vLLM args if needed

    args = parser.parse_args()

    logger.info("--- vLLM Launcher Starting ---")

    # 1. Get the key of the model that should be active
    active_model_key = get_active_model_key()
    if not active_model_key:
        logger.error("No active model key found. Cannot determine which model to launch.")
        # Fallback? Launch first model in config? Or exit? For now, exit.
        sys.exit("Launcher Error: Active model not specified.")

    # 2. Load the main model configuration
    model_config_data = load_json_config(MODEL_CONFIG_PATH)
    if not model_config_data:
        sys.exit(f"Launcher Error: Failed to load model config from {MODEL_CONFIG_PATH}")

    # 3. Find the config for the active model
    active_model_config = model_config_data.get(active_model_key)
    if not active_model_config:
        logger.error(f"Active model key '{active_model_key}' not found in configuration file {MODEL_CONFIG_PATH}.")
        sys.exit(f"Launcher Error: Configuration for active model '{active_model_key}' not found.")

    # 4. Extract necessary parameters
    model_id = active_model_config.get("model_id")
    if not model_id:
         sys.exit(f"Launcher Error: 'model_id' missing for active model '{active_model_key}'.")

    # Use model-specific TP size, fallback to 1 if not specified
    tensor_parallel_size = active_model_config.get("tensor_parallel_size", 1)
    # Use model-specific max length, fallback if needed
    max_model_len = active_model_config.get("max_model_len") # Let vLLM use its default if None
    # Use model-specific dtype, fallback if needed
    dtype = active_model_config.get("dtype") # Let vLLM use its default if None

    # Construct the full path to the model directory
    # vLLM's --model argument expects the HuggingFace ID or the *local path*
    # Using the local path is generally safer if downloads are managed separately
    model_path = os.path.join(MODELS_DIR, active_model_key)
    if not os.path.isdir(model_path):
         logger.error(f"Model directory for active model '{active_model_key}' not found at {model_path}. Has it been downloaded?")
         # Attempt to use model_id directly? Might fail if not cached by HF_HOME
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
    # Using subprocess is often more robust for managing external processes from Python
    vllm_command = [
        VENV_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--host", args.host,
        "--port", str(args.port),
        "--model", model_arg, # Use local path or HF ID
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        # Add other args conditionally
    ]
    if max_model_len:
        vllm_command.extend(["--max-model-len", str(max_model_len)])
    if dtype:
        vllm_command.extend(["--dtype", dtype])

    # Add any other necessary vLLM flags here, potentially from config
    # e.g., --quantization, --enable-lora, etc.

    logger.info(f"Executing command: {' '.join(vllm_command)}")

    # 6. Execute the vLLM server command
    # Use subprocess.run which waits for completion, or Popen for background
    # For systemd 'simple' service type, we want the process to stay in foreground
    try:
        # We replace the current process with vLLM using execvp for cleaner systemd handling
        # Ensure environment variables like HF_HOME are set by systemd if needed
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