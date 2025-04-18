import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.utils.gpu_utils import get_gpu_stats, get_gpu_count, is_apple_silicon
from backend.ray_deployments import build_llm_deployments

class TestNvidiaSpecific:
    """Test suite for NVIDIA-specific functionality."""
    
    def test_tensor_parallel_deployment(self):
        """Test that tensor parallel deployment works correctly with multiple GPUs."""
        # Skip on Apple Silicon environments
        if is_apple_silicon():
            pytest.skip("This test requires NVIDIA GPUs")
            
        # Get GPU count
        gpu_count = get_gpu_count()
        
        # Skip if less than 2 GPUs
        if gpu_count < 2:
            pytest.skip(f"This test requires at least 2 GPUs, but only {gpu_count} found")
        
        # Create a test config with tensor parallel size of 2
        config = {
            "test_model": {
                "model_id": "meta-llama/Llama-2-7b-chat-hf",
                "serve": True,
                "tensor_parallel_size": 2,  # Use 2 GPUs
                "max_model_len": 2048,
                "dtype": "float16"
            }
        }
        
        # Mock the necessary functions
        with patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=["config.json", "model.safetensors"]), \
             patch('backend.ray_deployments.build_llm_deployment') as mock_build_deployment, \
             patch('backend.ray_deployments.logger'):
            
            # Create a mock deployment
            mock_deployment = MagicMock()
            mock_build_deployment.return_value = mock_deployment
            
            # Build the deployments
            deployments = build_llm_deployments(config)
            
            # Verify the deployment was created
            assert len(deployments) == 1
            
            # Verify tensor_parallel_size was passed correctly
            args, kwargs = mock_build_deployment.call_args
            assert kwargs.get('tensor_parallel_size') == 2
    
    def test_cuda_device_detection(self):
        """Test that CUDA devices are correctly detected."""
        # Skip on Apple Silicon environments
        if is_apple_silicon():
            pytest.skip("This test requires NVIDIA GPUs")
        
        # Get GPU stats
        gpu_stats = get_gpu_stats()
        
        # Verify we have GPU stats
        assert gpu_stats is not None
        assert len(gpu_stats) > 0
        
        # Check that each GPU has the expected properties
        for gpu in gpu_stats:
            if hasattr(gpu, 'name'):
                # Object format
                assert 'NVIDIA' in gpu.name
                assert hasattr(gpu, 'memory_total_mb')
                assert hasattr(gpu, 'memory_used_mb')
                assert hasattr(gpu, 'gpu_utilization_pct')
            else:
                # Dict format
                assert 'NVIDIA' in gpu.get('name', '')
                assert 'memory_total_mb' in gpu
                assert 'memory_free_mb' in gpu or ('memory_total_mb' in gpu and 'memory_used_mb' in gpu)
                assert 'utilization' in gpu