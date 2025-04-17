import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# --- Configuration & Paths ---
VLLM_HOME = os.environ.get("VLLM_HOME", "/opt/vllm")
CONFIG_DIR = os.path.join(VLLM_HOME, "config")
MODELS_DIR = os.path.join(VLLM_HOME, "models")
LOGS_DIR = os.path.join(VLLM_HOME, "logs") # Central definition for log dir base

# --- Files ---
CONFIG_PATH = os.path.join(CONFIG_DIR, "model_config.json")
# ACTIVE_MODEL_FILE = os.path.join(CONFIG_DIR, "active_model.txt") # Obsolete with Ray Serve

# --- Service ---
# SERVICE_NAME = "vllm" # Obsolete with Ray Serve

# Directory creation is handled by start.sh in the target environment.
# os.makedirs(CONFIG_DIR, exist_ok=True) # Removed - causes issues during import/testing
# os.makedirs(MODELS_DIR, exist_ok=True) # Removed - causes issues during import/testing
# Log directory creation is handled in logging setup (main.py, app.py) using LOGS_DIR or VLLM_LOG_DIR env var


# --- Curated Models List ---
CURATED_MODELS_DATA: List[Dict[str, Any]] = [
    {
        "model_id": "meta-llama/Llama-3-8B-Instruct", "name": "Llama 3 (8B Instruct)", "size_gb": 16.0, "gated": True,
        "config": {"tensor_parallel_size": 1, "max_model_len": 8192, "dtype": "bfloat16"}
    },
    {
        "model_id": "meta-llama/Llama-3-70B-Instruct", "name": "Llama 3 (70B Instruct)", "size_gb": 140.0, "gated": True,
        "config": {"tensor_parallel_size": 2, "max_model_len": 8192, "dtype": "bfloat16"}
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2", "name": "Mistral (7B Instruct v0.2)", "size_gb": 14.0, "gated": False,
        "config": {"tensor_parallel_size": 1, "max_model_len": 8192, "dtype": "bfloat16"}
    },
    {
        "model_id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 (Mini 4K Instruct)", "size_gb": 9.0, "gated": False,
        "config": {"tensor_parallel_size": 1, "max_model_len": 4096, "dtype": "bfloat16"}
    },
    {
        "model_id": "NousResearch/Nous-Hermes-2-Yi-34B", "name": "Nous Hermes 2 Yi (34B)", "size_gb": 68.0, "gated": False,
        "config": {"tensor_parallel_size": 2, "max_model_len": 8192, "dtype": "bfloat16"}
    },
    {
        "model_id": "01-ai/Yi-1.5-34B-Chat", "name": "Yi 1.5 (34B Chat)", "size_gb": 68.0, "gated": False,
        "config": {"tensor_parallel_size": 2, "max_model_len": 8192, "dtype": "bfloat16"}
    },
    {
        "model_id": "01-ai/Yi-1.5-9B-Chat", "name": "Yi 1.5 (9B Chat)", "size_gb": 18.0, "gated": False,
        "config": {"tensor_parallel_size": 1, "max_model_len": 8192, "dtype": "bfloat16"}
    },
    {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama (1.1B Chat)", "size_gb": 2.2, "gated": False,
        "config": {"tensor_parallel_size": 1, "max_model_len": 2048, "dtype": "float16"}
    }
    # Add more curated models here if desired
]


# --- Config File Handling ---
def load_model_config() -> Dict[str, Any]:
    """Loads the model configuration from the JSON file."""
    if not os.path.exists(CONFIG_PATH):
        logger.warning(f"Config file not found: {CONFIG_PATH}. Returning empty config.")
        return {}
    try:
        with open(CONFIG_PATH, 'r') as f:
            content = f.read()
            if not content.strip(): # Handle empty file
                logger.warning(f"Config file {CONFIG_PATH} is empty. Returning empty config.")
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {CONFIG_PATH}. Returning empty config.")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file {CONFIG_PATH}: {e}")
        return {}

def save_model_config(config_data: Dict[str, Any]):
    """Saves the model configuration to the JSON file."""
    try:
        # Ensure directory exists before writing
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Model config saved successfully to {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error saving config file {CONFIG_PATH}: {e}")
        # Consider raising an exception for critical failures
        # raise IOError(f"Failed to save config file: {e}") from e

# Removed read_active_model_key() and write_active_model_key() as they are obsolete with Ray Serve.