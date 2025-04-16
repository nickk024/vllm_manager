import logging
import re
from huggingface_hub import HfApi
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Rough estimation factors (GB per Billion parameters)
VRAM_FACTOR_BF16_FP16 = 2.0
VRAM_FACTOR_INT8 = 1.2
VRAM_FACTOR_INT4 = 0.7
DEFAULT_VRAM_FACTOR = VRAM_FACTOR_BF16_FP16

def estimate_vram_gb(model_id: str, tags: List[str]) -> Optional[float]:
    """
    Very rough estimation of VRAM needed based on parameter count in model ID and tags.
    Returns estimated GB or None if size cannot be determined.
    """
    size_gb = None
    factor = DEFAULT_VRAM_FACTOR
    if "gguf" in tags or "awq" in tags or "gptq" in tags or any("4bit" in t for t in tags): factor = VRAM_FACTOR_INT4
    elif any("8bit" in t for t in tags): factor = VRAM_FACTOR_INT8

    match = re.search(r'([\d\.]+)[bB]', model_id)
    multimodal_match = re.search(r'(\d+)x(\d+)[bB]', model_id)

    if multimodal_match:
        try:
            num_experts = int(multimodal_match.group(1))
            params_b = float(multimodal_match.group(2))
            effective_params_b = params_b * 2.5 # Rough guess for MoE
            size_gb = effective_params_b * factor
            logger.debug(f"Estimated MoE model {model_id}: {num_experts}x{params_b}B -> Eff {effective_params_b:.1f}B -> {size_gb:.1f} GB (factor {factor})")
        except ValueError: logger.warning(f"Could not parse MoE params for {model_id}")
    elif match:
        try:
            params_b = float(match.group(1))
            size_gb = params_b * factor
            logger.debug(f"Estimated model {model_id}: {params_b}B -> {size_gb:.1f} GB (factor {factor})")
        except ValueError: logger.warning(f"Could not parse params for {model_id}")
    else: logger.debug(f"Could not determine parameter count for {model_id} from name.")

    if size_gb is not None: size_gb += 1.0 # Base overhead guess
    return size_gb


def fetch_dynamic_popular_models(
    available_vram_gb: Optional[float] = None,
    limit: int = 50, # Initial fetch limit
    top_n: int = 10, # Final return limit
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetches popular text-generation models from Hugging Face,
    estimates VRAM, filters by available VRAM, and returns the top N.
    """
    logger.info(f"Fetching dynamic popular models (fetch_limit={limit}, return_top_n={top_n}, available_vram={available_vram_gb} GB)")
    filtered_models = []
    try:
        api = HfApi(token=hf_token)
        # Fetch models - convert generator to list immediately
        models_iterator = api.list_models(
            sort="downloads",
            filter="text-generation",
            limit=limit * 2 # Fetch more initially
        )
        # Fix: Convert generator to list to get length and iterate
        all_fetched_models = list(models_iterator)
        logger.info(f"Fetched {len(all_fetched_models)} initial models from HF Hub.")

        count = 0
        for model_info in all_fetched_models:
            model_id = model_info.id
            tags = getattr(model_info, 'tags', [])

            if "gguf" in tags or "ggml" in tags:
                 logger.debug(f"Skipping GGUF/GGML model: {model_id}")
                 continue

            estimated_vram = estimate_vram_gb(model_id, tags)

            fits = True
            if available_vram_gb is not None and estimated_vram is not None:
                if estimated_vram > (available_vram_gb * 0.95): # Apply buffer
                    fits = False
                    logger.debug(f"Skipping model {model_id}: Est VRAM {estimated_vram:.1f} GB > Available {available_vram_gb:.1f} GB (w/ buffer)")

            if fits:
                default_config = {"tensor_parallel_size": 1, "max_model_len": 4096, "dtype": "bfloat16"}
                filtered_models.append({
                    "model_id": model_id,
                    "name": model_id,
                    "size_gb": round(estimated_vram, 1) if estimated_vram else 0.0,
                    "gated": getattr(model_info, 'gated', False),
                    "config": default_config,
                    "_debug_tags": tags,
                    "_debug_downloads": getattr(model_info, 'downloads', 0)
                })
                count += 1
                if count >= top_n:
                    logger.info(f"Reached top_n limit ({top_n}). Stopping processing.")
                    break # Stop once we have enough fitting models

    except Exception as e:
        logger.error(f"Failed to fetch or process models from Hugging Face Hub: {e}", exc_info=True)
        return []

    logger.info(f"Returning {len(filtered_models)} popular models potentially fitting VRAM.")
    return filtered_models