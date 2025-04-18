import pytest
from unittest.mock import patch, MagicMock
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from backend.ray_deployments import build_llm_deployments
from backend.routers.service_router import get_ray_serve_status


class TestConcurrency:
    """Test suite for concurrent operations and performance monitoring."""

    @pytest.mark.asyncio
    async def test_concurrent_model_loading(self):
        """Test that multiple models can be loaded concurrently."""
        # Create test config with multiple models
        config = {
            "model1": {
                "model_id": "test/model1",
                "serve": True,
                "tensor_parallel_size": 1,
                "max_model_len": 4096
            },
            "model2": {
                "model_id": "test/model2",
                "serve": True,
                "tensor_parallel_size": 1,
                "max_model_len": 4096
            },
            "model3": {
                "model_id": "test/model3",
                "serve": True,
                "tensor_parallel_size": 1,
                "max_model_len": 4096
            }
        }
        
        # Mock the necessary functions
        with patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=["config.json", "model.safetensors"]), \
             patch('backend.ray_deployments.build_llm_deployment') as mock_build_deployment, \
             patch('backend.ray_deployments.logger') as mock_logger, \
             patch('backend.ray_deployments.is_apple_silicon', return_value=False):
            
            # Create a mock deployment that takes some time to initialize
            def delayed_deployment(*args, **kwargs):
                time.sleep(0.1)  # Simulate some initialization time
                return MagicMock()
            
            mock_build_deployment.side_effect = delayed_deployment
            
            # Call the function to build deployments
            start_time = time.time()
            result = build_llm_deployments(config)
            end_time = time.time()
            
            # Verify that all models were deployed
            assert len(result) == 3
            
            # Verify that the deployments were built concurrently
            # If they were built sequentially, it would take at least 0.3 seconds
            # With concurrency, it should take less time
            # Allow for some overhead in the test environment (0.5 seconds)
            duration = end_time - start_time
            assert duration < 0.5, f"Models were not loaded concurrently (took {duration:.2f} seconds)"
            
            # Verify that the logger was called for each model
            assert mock_logger.info.call_count >= 3

    def test_ray_serve_status(self):
        """Test the Ray Serve status endpoint."""
        # Mock Ray and Ray Serve
        with patch('backend.routers.service_router.ray') as mock_ray, \
             patch('backend.routers.service_router.serve') as mock_serve:
            
            # Configure the mocks
            mock_ray.is_initialized.return_value = True
            mock_serve.status.return_value = {"status": "RUNNING"}
            
            # Call the function
            status = get_ray_serve_status()
            
            # Verify the result
            assert status["ray_initialized"] is True
            assert status["serve_running"] is True
            
            # Test when Ray is initialized but Serve is not running
            mock_serve.status.side_effect = Exception("Serve not running")
            status = get_ray_serve_status()
            assert status["ray_initialized"] is True
            assert status["serve_running"] is False
            
            # Test when Ray is not initialized
            mock_ray.is_initialized.return_value = False
            status = get_ray_serve_status()
            assert status["ray_initialized"] is False
            assert status["serve_running"] is False

    def test_concurrent_status_requests(self):
        """Test handling of concurrent status requests."""
        # Mock Ray and Ray Serve
        with patch('backend.routers.service_router.ray') as mock_ray, \
             patch('backend.routers.service_router.serve') as mock_serve:
            
            # Configure the mocks
            mock_ray.is_initialized.return_value = True
            mock_serve.status.return_value = {"status": "RUNNING"}
            
            # Function to get status in a thread
            def get_status():
                return get_ray_serve_status()
            
            # Create a thread pool and submit multiple requests
            results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(get_status) for _ in range(10)]
                for future in futures:
                    results.append(future.result())
            
            # Verify all results are consistent
            for result in results:
                assert result["ray_initialized"] is True
                assert result["serve_running"] is True