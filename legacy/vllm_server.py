#!/usr/bin/env python3
"""
vLLM API server script with dynamic model loading capability
"""
import os
import json
import argparse
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vllm_server")

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM API Server")
    parser.add_argument("--model-dir", type=str, help="Directory containing model directories")
    parser.add_argument("--config", type=str, help="Path to model configuration JSON")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Tensor parallelism size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum sequence length")
    return parser.parse_args()

def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from JSON file"""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, "r") as f:
        return json.load(f)

def start_server(args):
    """Start the vLLM OpenAI-compatible API server"""
    from vllm.entrypoints.openai.api_server import main
    import sys
    
    # Prepare command-line arguments for vLLM
    cmd_args = [
        "api_server.py",
        "--host", args.host,
        "--port", str(args.port),
        "--model-dir", args.model_dir,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len)
    ]
    
    # Override sys.argv with our args
    sys.argv = cmd_args
    
    # Start the server
    logger.info(f"Starting vLLM server on {args.host}:{args.port}")
    logger.info(f"Using model directory: {args.model_dir}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Error starting vLLM server: {e}")
        raise

def prepare_models(model_dir: str, config_path: str):
    """Ensure model directories exist with config files"""
    if not os.path.exists(config_path):
        logger.warning(f"Model config not found: {config_path}")
        return
    
    config = load_model_config(config_path)
    if not config:
        logger.warning("Empty or invalid model configuration")
        return
    
    for model_name, model_config in config.items():
        model_path = os.path.join(model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Create model config.json file
        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Prepared configuration for model: {model_name}")

def main():
    args = parse_args()
    
    # Prepare model directories
    prepare_models(args.model_dir, args.config)
    
    # Start the server
    start_server(args)

if __name__ == "__main__":
    main()
