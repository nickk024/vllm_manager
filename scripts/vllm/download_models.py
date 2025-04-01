#!/usr/bin/env python3
"""
Model downloader for vLLM
"""
import os
import json
import logging
import argparse
from huggingface_hub import snapshot_download, HfApi
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_downloader")

def parse_args():
    parser = argparse.ArgumentParser(description="Download models for vLLM")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model configuration JSON")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to store downloaded models")
    parser.add_argument("--models", type=str, nargs="+", 
                        help="Specific models to download (default: all in config)")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token for downloading gated models")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if model exists")
    return parser.parse_args()

def load_model_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def download_model(model_name, model_config, output_dir, token=None, force=False):
    """Download a model and prepare it for vLLM."""
    model_dir = os.path.join(output_dir, model_name)
    
    # Check if model already exists
    if not force and os.path.exists(model_dir) and os.listdir(model_dir):
        logger.info(f"Model {model_name} already exists at {model_dir}")
        
        # Create/update config.json for vLLM
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
            
        return True
    
    model_id = model_config["model_id"]
    revision = model_config.get("revision", "main")
    
    # Make sure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Download the model
        logger.info(f"Downloading {model_id} (revision: {revision}) to {model_dir}")
        
        # Check if token is required
        if "meta-llama" in model_id.lower() and not token:
            logger.error(f"Hugging Face token required for {model_id}. Set HF_TOKEN environment variable.")
            return False
        
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=model_dir,
            token=token
        )
        
        # Create config.json for vLLM
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Successfully downloaded {model_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        return False

def main():
    args = parse_args()
    
    # Load model configs
    model_configs = load_model_config(args.config)
    
    # Filter to requested models if specified
    if args.models:
        model_configs = {name: config for name, config in model_configs.items() 
                         if name in args.models}
    
    if not model_configs:
        logger.error("No models found in configuration or specified models not found.")
        return
    
    # Download models
    success_count = 0
    total_models = len(model_configs)
    
    logger.info(f"Preparing to download {total_models} model(s)")
    
    for i, (model_name, config) in enumerate(model_configs.items(), 1):
        logger.info(f"[{i}/{total_models}] Processing model: {model_name}")
        if download_model(model_name, config, args.output_dir, args.token, args.force):
            success_count += 1
    
    logger.info(f"Downloaded {success_count}/{total_models} models successfully")
    
    if success_count < total_models:
        logger.warning("Some models failed to download. Check the logs for details.")
    else:
        logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    main()
