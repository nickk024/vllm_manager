import logging
import re
from huggingface_hub import HfApi
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Rough estimation factors (GB per Billion parameters)
# These are VERY approximate and depend heavily on quantization, context length etc.
# BF16/FP16 usually takes ~2 bytes per parameter.
VRAM_FACTOR_BF16_FP16 = 2.0
# INT8 might be ~1 byte per parameter + overhead
VRAM_FACTOR_INT8 = 1.2
# INT4 might be ~0.5 bytes per parameter + overhead
VRAM_FACTOR_INT4 = 0.7

# Default factor if quantization is unknown (assume bf16/fp16)
DEFAULT_VRAM_FACTOR = VRAM_FACTOR_BF16_FP16

def estimate_vram_gb(model_id: str, tags: List[str]) -> Optional[float]:
    """
    Very rough estimation of VRAM needed based on parameter count in model ID and tags.
    Returns estimated GB or None if size cannot be determined.
    """
    size_gb = None
    factor = DEFAULT_VRAM_FACTOR

    # Check tags for quantization info first
    if "gguf" in tags or "awq" in tags or "gptq" in tags or any("4bit" in t for t in tags):
        factor = VRAM_FACTOR_INT4 # Assume 4-bit if specific quant tag found
    elif any("8bit" in t for t in tags):
         factor = VRAM_FACTOR_INT8

    # Try to extract parameter count (e.g., 7b, 13b, 70b, 8x7b) from model_id
    match = re.search(r'([\d\.]+)[bB]', model_id) # Matches 7b, 13B, 1.1B etc.
    multimodal_match = re.search(r'(\d+)x(\d+)[bB]', model_id) # Matches 8x7b etc.

    if multimodal_match:
        try:
            num_experts = int(multimodal_match.group(1))
            params_b = float(multimodal_match.group(2))
            # Rough estimate for MoE models (often only a subset of experts active)
            # This is highly variable, let's estimate based on ~2 active experts?
            effective_params_b = params_b * 2.5 # Very rough guess
            size_gb = effective_params_b * factor
            logger.debug(f"Estimated MoE model {model_id}: {num_experts}x{params_b}B -> Effective {effective_params_b:.1f}B -> {size_gb:.1f} GB (factor {factor})")
        except ValueError:
            logger.warning(f"Could not parse MoE params for {model_id}")
    elif match:
        try:
            params_b = float(match.group(1))
            size_gb = params_b * factor
            logger.debug(f"Estimated model {model_id}: {params_b}B -> {size_gb:.1f} GB (factor {factor})")
        except ValueError:
            logger.warning(f"Could not parse params for {model_id}")
    else:
         logger.debug(f"Could not determine parameter count for {model_id} from name.")

    # Add a small base overhead (e.g., for KV cache, framework) - highly variable!
    if size_gb is not None:
        size_gb += 1.0 # Add 1GB base overhead guess

    return size_gb


def fetch_dynamic_popular_models(
    available_vram_gb: Optional[float] = None,
    limit: int = 50,
    top_n: int = 10,
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetches popular text-generation models from Hugging Face,
    estimates VRAM, filters by available VRAM, and returns the top N.
    """
    logger.info(f"Fetching dynamic popular models (limit={limit}, top_n={top_n}, available_vram={available_vram_gb} GB)")
    filtered_models = []
    try:
        api = HfApi(token=hf_token)
        # Fetch models sorted by downloads, filtering for text-generation and excluding GGUF explicitly for vLLM base
        # Add more filters? e.g., -gguf? Or rely on vllm compatibility later?
        models = api.list_models(
            sort="downloads",
            filter="text-generation", # Primary filter
            limit=limit * 2 # Fetch more initially to allow for filtering
        )
        logger.info(f"Fetched {len(models)} initial models from HF Hub.")

        count = 0
        for model_info in models:
            model_id = model_info.id
            tags = getattr(model_info, 'tags', [])

            # Basic filtering (can be expanded)
            if "vllm" in tags: # Prioritize models explicitly tagged for vLLM
                 pass # Good sign
            if "gguf" in tags or "ggml" in tags: # Exclude GGUF/GGML formats
                 logger.debug(f"Skipping GGUF/GGML model: {model_id}")
                 continue
            # Add other exclusion criteria? e.g., based on license?

            estimated_vram = estimate_vram_gb(model_id, tags)

            # Check if it fits available VRAM if specified
            fits = True # Assume fits if no VRAM provided or estimation fails
            if available_vram_gb is not None and estimated_vram is not None:
                # Add a small buffer (e.g., 10%) to available VRAM for safety
                if estimated_vram > (available_vram_gb * 0.95):
                    fits = False
                    logger.debug(f"Skipping model {model_id}: Estimated VRAM {estimated_vram:.1f} GB > Available {available_vram_gb:.1f} GB (with buffer)")

            if fits:
                # Use default config for now, could potentially fetch actual config later if needed
                default_config = {"tensor_parallel_size": 1, "max_model_len": 4096, "dtype": "bfloat16"}
                filtered_models.append({
                    "model_id": model_id,
                    "name": model_id, # Use ID as name for simplicity, could refine
                    "size_gb": round(estimated_vram, 1) if estimated_vram else 0.0, # Store estimated size
                    "gated": getattr(model_info, 'gated', False),
                    "config": default_config, # Use a default base config
                    "_debug_tags": tags, # Include tags for debugging/display
                    "_debug_downloads": getattr(model_info, 'downloads', 0) # Include downloads for sorting check
                })
                count += 1
                if count >= top_n:
                    break # Stop once we have enough fitting models

    except Exception as e:
        logger.error(f"Failed to fetch or process models from Hugging Face Hub: {e}", exc_info=True)
        # Fallback or return empty? Return empty for now.
        return []

    logger.info(f"Returning {len(filtered_models)} popular models potentially fitting VRAM.")
    return filtered_models