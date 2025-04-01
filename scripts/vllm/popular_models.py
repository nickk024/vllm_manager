#!/usr/bin/env python3
"""
Script to fetch popular LLM models from Hugging Face and present them for selection
"""
import os
import sys
import json
import logging
import argparse
import requests
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("popular_models")

# ANSI color codes for terminal output
COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "BOLD": "\033[1m",
    "RESET": "\033[0m",
}

# List of popular model categories that work with vLLM
MODEL_CATEGORIES = [
    {
        "name": "Open Large Language Models",
        "id": "llm-open",
        "description": "Models like Llama, Mistral, Falcon, etc."
    },
    {
        "name": "Instruction Fine-tuned Models",
        "id": "llm-instruct",
        "description": "Models specifically tuned for instructions/chat"
    },
    {
        "name": "Small, Efficient Models",
        "id": "llm-small",
        "description": "Smaller models like Phi, TinyLlama, etc."
    }
]

# Curated list of known popular models that work well with vLLM
CURATED_MODELS = [
    {
        "model_id": "meta-llama/Llama-3-8B-Instruct",
        "name": "Llama 3 (8B Instruct)",
        "description": "Meta's Llama 3 model, 8B parameter version with instruction tuning",
        "size_gb": 16.0,
        "gated": True,
        "category": "llm-open",
        "config": {
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
            "dtype": "bfloat16"
        }
    },
    {
        "model_id": "meta-llama/Llama-3-70B-Instruct",
        "name": "Llama 3 (70B Instruct)",
        "description": "Meta's Llama 3 flagship model, 70B parameter version with instruction tuning. Requires multiple GPUs.",
        "size_gb": 140.0,
        "gated": True,
        "category": "llm-open",
        "config": {
            "tensor_parallel_size": 2,
            "max_model_len": 8192,
            "dtype": "bfloat16"
        }
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "name": "Mistral (7B Instruct v0.2)",
        "description": "Mistral's 7B parameter model with instruction tuning",
        "size_gb": 14.0,
        "gated": False,
        "category": "llm-open",
        "config": {
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
            "dtype": "bfloat16"
        }
    },
    {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "name": "Phi-3 (Mini 4K Instruct)",
        "description": "Microsoft's small but powerful Phi-3 model",
        "size_gb": 9.0,
        "gated": False,
        "category": "llm-small",
        "config": {
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "dtype": "bfloat16"
        }
    },
    {
        "model_id": "NousResearch/Nous-Hermes-2-Yi-34B",
        "name": "Nous Hermes 2 Yi (34B)",
        "description": "Nous Research's 34B parameter model based on Yi",
        "size_gb": 68.0,
        "gated": False,
        "category": "llm-instruct",
        "config": {
            "tensor_parallel_size": 2,
            "max_model_len": 8192,
            "dtype": "bfloat16"
        }
    },
    {
        "model_id": "01-ai/Yi-1.5-34B-Chat",
        "name": "Yi 1.5 (34B Chat)",
        "description": "Zero-One AI's powerful 34B chat model",
        "size_gb": 68.0,
        "gated": False,
        "category": "llm-instruct",
        "config": {
            "tensor_parallel_size": 2,
            "max_model_len": 8192,
            "dtype": "bfloat16"
        }
    },
    {
        "model_id": "01-ai/Yi-1.5-9B-Chat",
        "name": "Yi 1.5 (9B Chat)",
        "description": "Zero-One AI's efficient 9B chat model",
        "size_gb": 18.0,
        "gated": False,
        "category": "llm-instruct",
        "config": {
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
            "dtype": "bfloat16"
        }
    },
    {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama (1.1B Chat)",
        "description": "Very small, efficient LLM for resource-constrained environments",
        "size_gb": 2.2,
        "gated": False,
        "category": "llm-small",
        "config": {
            "tensor_parallel_size": 1,
            "max_model_len": 2048,
            "dtype": "float16"
        }
    }
]

def fetch_popular_models(api_token: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """
    Fetch popular LLM models from Hugging Face API
    """
    logger.info("Fetching popular models from Hugging Face...")
    
    # We'll use a combination of our curated list and real-time HF API
    models = CURATED_MODELS.copy()
    
    # Optional: Use the HF API to get trending models
    # This part requires an API token and Hugging Face rate limits can be strict
    try:
        if api_token:
            # For the API endpoint, see HF docs: https://huggingface.co/docs/hub/api
            hf_api_url = "https://huggingface.co/api/models?sort=trending&direction=-1&filter=text-generation&limit=20"
            headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
            
            response = requests.get(hf_api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                hf_models = response.json()
                
                # Convert HF API response to our format
                for model in hf_models:
                    # Skip models already in our curated list
                    if any(m["model_id"] == model["id"] for m in models):
                        continue
                    
                    # Only add compatible models that can work with vLLM
                    # This is a simplified check - in a real world scenario this would be more complex
                    if "text-generation" in model.get("pipeline_tag", []):
                        models.append({
                            "model_id": model["id"],
                            "name": model.get("name", model["id"].split("/")[-1]),
                            "description": model.get("description", "No description available").split("\n")[0][:100],
                            "size_gb": 0.0,  # Unknown size
                            "gated": model.get("gated", False),
                            "category": "llm-open",  # Default category
                            "config": {
                                "tensor_parallel_size": 1,
                                "max_model_len": 4096,
                                "dtype": "bfloat16"
                            }
                        })
                        
                        # Only add up to the limit
                        if len(models) >= limit:
                            break
                
                logger.info(f"Fetched {len(hf_models)} models from Hugging Face API")
            else:
                logger.warning(f"Failed to fetch from HF API: {response.status_code}")
                
    except Exception as e:
        logger.warning(f"Error fetching from Hugging Face API: {e}")
        logger.info("Proceeding with curated model list only")
    
    # Sort models by name
    models.sort(key=lambda x: x["name"])
    
    return models

def display_model_list(models: List[Dict], categories: List[Dict]) -> None:
    """
    Display the list of models in a formatted way
    """
    # Group models by category
    models_by_category = {}
    for category in categories:
        models_by_category[category["id"]] = []
    
    for model in models:
        category = model.get("category", "llm-open")
        if category in models_by_category:
            models_by_category[category].append(model)
    
    # Print models by category
    for category in categories:
        category_id = category["id"]
        if models_by_category[category_id]:
            print(f"\n{COLORS['BOLD']}{COLORS['CYAN']}=== {category['name']} ==={COLORS['RESET']}")
            print(f"{category['description']}")
            print("\nAvailable models:")
            
            for i, model in enumerate(models_by_category[category_id], 1):
                gated_marker = f" {COLORS['YELLOW']}[Gated]{COLORS['RESET']}" if model["gated"] else ""
                size_info = f" (~{model['size_gb']:.1f}GB)" if model["size_gb"] > 0 else ""
                print(f"  {i}. {COLORS['GREEN']}{model['name']}{COLORS['RESET']}{gated_marker}{size_info}")
                print(f"     {model['description']}")
                print(f"     ID: {model['model_id']}")
                print()

def select_models(models: List[Dict]) -> List[Dict]:
    """
    Interactive prompt for the user to select models
    """
    print(f"\n{COLORS['BOLD']}Select models to download (comma-separated numbers, empty to cancel):{COLORS['RESET']}")
    
    # First display the models by category
    display_model_list(models, MODEL_CATEGORIES)
    
    # Create a flat list of all models for selection
    all_models = []
    for category in MODEL_CATEGORIES:
        for model in models:
            if model.get("category") == category["id"]:
                all_models.append(model)
    
    # Show a simple numbered list for selection
    print(f"\n{COLORS['BOLD']}All Models:{COLORS['RESET']}")
    for i, model in enumerate(all_models, 1):
        gated_marker = f" {COLORS['YELLOW']}[Gated]{COLORS['RESET']}" if model["gated"] else ""
        size_info = f" (~{model['size_gb']:.1f}GB)" if model["size_gb"] > 0 else ""
        print(f"  {i}. {model['name']}{gated_marker}{size_info}")
    
    # Get user selection
    try:
        selection = input("\nEnter your choices (e.g., 1,3,4): ").strip()
        if not selection:
            return []
        
        indices = [int(idx.strip()) - 1 for idx in selection.split(",") if idx.strip()]
        selected_models = [all_models[i] for i in indices if 0 <= i < len(all_models)]
        
        return selected_models
    except (ValueError, IndexError) as e:
        print(f"{COLORS['RED']}Invalid selection: {e}{COLORS['RESET']}")
        return []

def generate_vllm_config(selected_models: List[Dict], output_file: str, gpu_count: int = 1) -> None:
    """
    Generate a vLLM configuration file for the selected models
    """
    # Initialize the config dict
    config = {}
    
    # Add each selected model to the config
    for model in selected_models:
        model_name = model["model_id"].split("/")[-1].lower().replace("-", "_")
        
        # Adjust tensor parallel size based on available GPUs and model size
        tensor_parallel_size = model["config"]["tensor_parallel_size"]
        if model["size_gb"] > 30 and gpu_count > 1:
            tensor_parallel_size = min(tensor_parallel_size, gpu_count)
        elif model["size_gb"] <= 10:
            tensor_parallel_size = 1
        
        config[model_name] = {
            "model_id": model["model_id"],
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": model["config"]["max_model_len"],
            "dtype": model["config"]["dtype"]
        }
    
    # Write the config to file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Generated vLLM config for {len(selected_models)} models at {output_file}")
    
    # Also return the config as a string
    return json.dumps(config, indent=4)

def get_gpu_count() -> int:
    """Get number of available NVIDIA GPUs"""
    try:
        import torch
        return torch.cuda.device_count()
    except:
        try:
            # Try using nvidia-smi as fallback
            import subprocess
            output = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode()
            return len(output.strip().split("\n"))
        except:
            return 1

def prepare_download_command(selected_models: List[Dict], config_file: str, output_dir: str, token: Optional[str] = None) -> str:
    """
    Prepare a download command for the selected models
    """
    command = f"python download_models.py --config {config_file} --output-dir {output_dir}"
    
    if token:
        command += f" --token {token}"
    
    return command

def main():
    parser = argparse.ArgumentParser(description="Fetch and select popular LLM models for vLLM")
    parser.add_argument("--output", type=str, default="model_config.json", help="Output configuration file")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to store downloaded models")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    parser.add_argument("--generate-only", action="store_true", help="Only generate config without interactive selection")
    parser.add_argument("--list-only", action="store_true", help="Only list models without interactive selection")
    args = parser.parse_args()
    
    # Fetch popular models
    models = fetch_popular_models(args.token)
    
    if args.list_only:
        # Just display the models without selection
        display_model_list(models, MODEL_CATEGORIES)
        return 0
    
    if args.generate_only:
        # Generate config with all models
        gpu_count = get_gpu_count()
        
        logger.info(f"Detected {gpu_count} GPUs")
        config = generate_vllm_config(models, args.output, gpu_count)
        
        print(f"\n{COLORS['BOLD']}{COLORS['GREEN']}Configuration generated with all models:{COLORS['RESET']}")
        print(f"Output file: {args.output}")
        print(f"Models included: {len(models)}")
        
        return 0
    
    # Interactive selection mode
    selected_models = select_models(models)
    
    if not selected_models:
        print(f"{COLORS['YELLOW']}No models selected. Exiting.{COLORS['RESET']}")
        return 0
    
    # Print summary
    print(f"\n{COLORS['BOLD']}Selected models:{COLORS['RESET']}")
    total_size = 0
    gated_models = []
    for model in selected_models:
        gated_marker = f" {COLORS['YELLOW']}[Gated]{COLORS['RESET']}" if model["gated"] else ""
        size_info = f" (~{model['size_gb']:.1f}GB)" if model["size_gb"] > 0 else ""
        print(f"  - {model['name']}{gated_marker}{size_info}")
        total_size += model["size_gb"]
        if model["gated"]:
            gated_models.append(model["name"])
    
    print(f"\nTotal estimated download size: ~{total_size:.1f} GB")
    
    # Warning about gated models
    if gated_models and not args.token:
        print(f"\n{COLORS['YELLOW']}Warning: You've selected gated models ({', '.join(gated_models)}) but no token was provided.")
        print(f"You'll need a Hugging Face token to download these models.{COLORS['RESET']}")
        token_input = input("Enter your Hugging Face token (or press Enter to continue without token): ").strip()
        if token_input:
            args.token = token_input
    
    # Generate config
    gpu_count = get_gpu_count()
    logger.info(f"Detected {gpu_count} GPUs")
    config = generate_vllm_config(selected_models, args.output, gpu_count)
    
    # Print download command
    download_cmd = prepare_download_command(selected_models, args.output, args.models_dir, args.token)
    
    print(f"\n{COLORS['BOLD']}{COLORS['GREEN']}Configuration generated successfully!{COLORS['RESET']}")
    print(f"Output file: {args.output}")
    
    print(f"\n{COLORS['BOLD']}To download the selected models, run:{COLORS['RESET']}")
    print(f"  {download_cmd}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
