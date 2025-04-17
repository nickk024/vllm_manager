import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from backend.utils.hf_utils import estimate_vram_gb, fetch_dynamic_popular_models

class TestHfUtils:
    """Test suite for Hugging Face utilities."""
    
    @pytest.mark.parametrize("model_id, tags, expected", [
        # Standard models with different parameter sizes
        ("meta-llama/Llama-3-8B-Instruct", [], 8.0 * 2.0 + 1.0),  # 8B model with default factor
        ("mistralai/Mistral-7B-Instruct-v0.2", [], 7.0 * 2.0 + 1.0),  # 7B model with default factor
        ("microsoft/Phi-3-mini-4k-instruct", [], None),  # No parameter count in name
        
        # Models with quantization tags
        ("TheBloke/Llama-2-7B-GGUF", ["gguf"], 7.0 * 0.7 + 1.0),  # GGUF model with INT4 factor
        ("TheBloke/Llama-2-7B-GPTQ", ["gptq"], 7.0 * 0.7 + 1.0),  # GPTQ model with INT4 factor
        ("TheBloke/Llama-2-7B-AWQ", ["awq"], 7.0 * 0.7 + 1.0),  # AWQ model with INT4 factor
        ("TheBloke/Llama-2-7B-8bit", ["8bit"], 7.0 * 1.2 + 1.0),  # 8-bit model with INT8 factor
        
        # MoE models
        ("mistralai/Mixtral-8x7B-v0.1", [], 7.0 * 2.5 * 2.0 + 1.0),  # MoE model with 8 experts
        
        # Edge cases
        ("model-without-parameter-count", [], None),  # No parameter count in name
        ("", [], None),  # Empty model ID
    ])
    def test_estimate_vram_gb(self, model_id: str, tags: List[str], expected: float):
        """Test VRAM estimation for different model types."""
        result = estimate_vram_gb(model_id, tags)
        
        if expected is None:
            assert result is None
        else:
            assert result is not None
            assert abs(result - expected) < 0.1  # Allow small floating point differences
    
    def test_fetch_dynamic_popular_models_success(self):
        """Test successful fetching of popular models."""
        # Create mock model objects
        mock_models = [
            MagicMock(id="meta-llama/Llama-3-8B-Instruct", tags=["text-generation"], downloads=10000, gated=True),
            MagicMock(id="mistralai/Mistral-7B-Instruct-v0.2", tags=["text-generation"], downloads=8000, gated=False),
            MagicMock(id="microsoft/Phi-3-mini-4k-instruct", tags=["text-generation"], downloads=6000, gated=False),
            MagicMock(id="TheBloke/Llama-2-7B-GGUF", tags=["text-generation", "gguf"], downloads=5000, gated=False),
        ]
        
        # Mock the HfApi class and its instance
        with patch('backend.utils.hf_utils.HfApi') as mock_hf_api_class:
            # Configure the mock API instance
            mock_api_instance = MagicMock()
            mock_hf_api_class.return_value = mock_api_instance
            mock_api_instance.list_models.return_value = mock_models
            
            # Call the function with no VRAM limit
            result = fetch_dynamic_popular_models(top_n=3)
            
            # Verify the results
            assert len(result) == 3  # Should return top 3 models
            assert result[0]["model_id"] == "meta-llama/Llama-3-8B-Instruct"
            assert result[1]["model_id"] == "mistralai/Mistral-7B-Instruct-v0.2"
            assert result[2]["model_id"] == "microsoft/Phi-3-mini-4k-instruct"
            
            # Verify GGUF model was skipped
            assert not any(m["model_id"] == "TheBloke/Llama-2-7B-GGUF" for m in result)
    
    def test_fetch_dynamic_popular_models_with_vram_filter(self):
        """Test fetching popular models with VRAM filtering."""
        # Create mock model objects
        mock_models = [
            MagicMock(id="meta-llama/Llama-3-70B-Instruct", tags=["text-generation"], downloads=10000, gated=True),
            MagicMock(id="mistralai/Mistral-7B-Instruct-v0.2", tags=["text-generation"], downloads=8000, gated=False),
            MagicMock(id="microsoft/Phi-3-mini-4k-instruct", tags=["text-generation"], downloads=6000, gated=False),
        ]
        
        # Mock the HfApi class and its instance
        with patch('backend.utils.hf_utils.HfApi') as mock_hf_api_class:
            # Configure the mock API instance
            mock_api_instance = MagicMock()
            mock_hf_api_class.return_value = mock_api_instance
            mock_api_instance.list_models.return_value = mock_models
            
            # Call the function with a VRAM limit that excludes the 70B model
            result = fetch_dynamic_popular_models(available_vram_gb=24.0, top_n=3)
            
            # Verify the results
            assert len(result) == 2  # Should return only models that fit
            assert result[0]["model_id"] == "mistralai/Mistral-7B-Instruct-v0.2"
            assert result[1]["model_id"] == "microsoft/Phi-3-mini-4k-instruct"
            
            # Verify 70B model was filtered out due to VRAM constraint
            assert not any(m["model_id"] == "meta-llama/Llama-3-70B-Instruct" for m in result)
    
    def test_fetch_dynamic_popular_models_api_error(self):
        """Test error handling when the HF API fails."""
        # Mock the HfApi to raise an exception
        with patch('backend.utils.hf_utils.HfApi') as mock_hf_api:
            mock_api_instance = mock_hf_api.return_value
            mock_api_instance.list_models.side_effect = Exception("API Error")
            
            # Call the function
            result = fetch_dynamic_popular_models()
            
            # Verify the results
            assert result == []  # Should return empty list on error
    
    def test_fetch_dynamic_popular_models_with_token(self):
        """Test fetching popular models with an HF token."""
        # Create mock model objects
        mock_models = [MagicMock(id="meta-llama/Llama-3-8B-Instruct", tags=["text-generation"], downloads=10000, gated=True)]
        
        # Mock the HfApi class and its instance
        with patch('backend.utils.hf_utils.HfApi') as mock_hf_api_class:
            # Configure the mock API instance
            mock_api_instance = MagicMock()
            mock_hf_api_class.return_value = mock_api_instance
            mock_api_instance.list_models.return_value = mock_models
            
            # Call the function with a token
            result = fetch_dynamic_popular_models(hf_token="test_token")
            
            # Verify the HfApi was initialized with the token
            mock_hf_api_class.assert_called_once_with(token="test_token")