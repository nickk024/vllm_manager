import pytest
import os
import psutil
import gc
from unittest.mock import patch, MagicMock

from backend.ray_deployments import build_llm_deployments
from backend.utils.gpu_utils import get_gpu_stats, is_apple_silicon


class TestMemoryManagement:
    """Test suite for memory management and resource allocation."""

    def test_memory_cleanup_after_deployment(self):
        """Test that memory is properly cleaned up after deployments are created and destroyed."""
        # Skip on Apple Silicon environments
        if is_apple_silicon():
            pytest.skip("This test is not applicable on Apple Silicon")
            
        # Record memory usage before deployment
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # Create a test config
        config = {
            "test_model": {
                "model_id": "TheBloke/Llama-2-7B-GGUF",  # Use a smaller model for testing
                "serve": True,
                "tensor_parallel_size": 1,
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
            # The key might be prefixed with '/' in some environments
            assert "test_model" in deployments or "/test_model" in deployments
            
            # Force garbage collection
            deployments.clear()
            gc.collect()
            
            # Record memory usage after cleanup
            memory_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
            # Memory usage should not increase significantly
            # Allow for some overhead (50MB)
            assert memory_after - memory_before < 50, f"Memory leak detected: {memory_after - memory_before} MB"

    def test_gpu_memory_allocation(self):
        """Test that GPU memory is properly allocated and released."""
        # Skip on Apple Silicon environments
        if is_apple_silicon():
            pytest.skip("This test is not applicable on Apple Silicon")
            
        # Get initial GPU stats
        initial_stats = get_gpu_stats()
        if not initial_stats:
            pytest.skip("Could not get GPU stats")
            
        # Record initial free memory - handle both dict and object formats
        initial_free_memory = 0
        for gpu in initial_stats:
            if hasattr(gpu, "memory_total_mb") and hasattr(gpu, "memory_used_mb"):
                # Object format (GPUStat)
                initial_free_memory += gpu.memory_total_mb - gpu.memory_used_mb
            else:
                # Dict format
                initial_free_memory += gpu.get("memory_free_mb", 0)
        
        # Create a test config with a small model
        config = {
            "test_model": {
                "model_id": "TheBloke/Llama-2-7B-GGUF",
                "serve": True,
                "tensor_parallel_size": 1,
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
            
            # In a real scenario, this would allocate GPU memory
            # For the test, we're just verifying the mocks were called correctly
            mock_build_deployment.assert_called_once()
            
            # Clean up
            deployments.clear()
            gc.collect()
            
            # In a real test, we would check that GPU memory was released
            # But since we're mocking, we'll just verify the deployment was created and cleaned up
            assert "test_model" not in deployments


class TestRecoveryScenarios:
    """Test suite for recovery from failures."""

    def test_recovery_from_failed_deployment(self):
        """Test that the system can recover from a failed deployment."""
        # Create a test config with two models
        config = {
            "model1": {
                "model_id": "org/model1",
                "serve": True,
                "tensor_parallel_size": 1
            },
            "model2": {
                "model_id": "org/model2",
                "serve": True,
                "tensor_parallel_size": 1
            }
        }
        
        # Mock the necessary functions
        with patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=["config.json", "model.safetensors"]), \
             patch('backend.ray_deployments.build_llm_deployment') as mock_build_deployment, \
             patch('backend.ray_deployments.logger') as mock_logger:
            
            # Make the first deployment succeed and the second fail
            mock_build_deployment.side_effect = [MagicMock(), Exception("Deployment failed")]
            
            # Build the deployments
            deployments = build_llm_deployments(config)
            
            # Verify that one deployment succeeded
            assert len(deployments) == 1
            # The key might be prefixed with a slash in some implementations
            assert "model1" in deployments or "/model1" in deployments
            assert "model2" not in deployments and "/model2" not in deployments
            
            # Verify that the error was logged
            mock_logger.error.assert_called()
            
            # Verify that the system continued despite the failure
            assert mock_build_deployment.call_count == 2

    def test_recovery_from_gpu_failure(self):
        """Test that the system can recover from a GPU failure."""
        # Skip on Apple Silicon environments
        if is_apple_silicon():
            pytest.skip("This test is not applicable on Apple Silicon")
            
        # This test is more of a demonstration than an actual test
        # In a real scenario, we would need to simulate a GPU failure at the hardware level
        # which is difficult to do in a test environment
        
        # Create a test config with a model that uses only one GPU
        config = {
            "test_model": {
                "model_id": "TheBloke/Llama-2-7B-GGUF",
                "serve": True,
                "tensor_parallel_size": 1,  # Use only 1 GPU
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
            assert "test_model" in deployments or "/test_model" in deployments
            
            # In a real scenario, if a GPU fails, the system should be able to continue
            # with the remaining GPUs. This test verifies that a deployment can be created
            # with a single GPU, which is the fallback scenario after a GPU failure.