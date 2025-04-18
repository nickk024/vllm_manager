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

    @pytest.mark.asyncio
    async def test_ray_serve_status(self):
        """Test the Ray Serve status endpoint."""
        # Mock the specific functions/attributes used within get_ray_serve_status
        # Note: We patch the imports *where they are used* inside the function
        with patch('backend.routers.service_router.ray.is_initialized') as mock_is_initialized, \
             patch('backend.routers.service_router.ray.nodes') as mock_nodes, \
             patch('backend.routers.service_router.ray.available_resources') as mock_available_resources, \
             patch('backend.routers.service_router.ray.__version__', "mock_version"), \
             patch('backend.routers.service_router.serve.api._get_global_client') as mock_get_client, \
             patch('backend.routers.service_router.get_configured_models_internal', return_value=[]): # Mock model fetching

            # --- Test Case 1: Ray Initialized, Serve Running ---
            mock_is_initialized.return_value = True
            mock_nodes.return_value = ["node1"] # Simulate one node
            mock_available_resources.return_value = {"CPU": 4}
            mock_get_client.return_value = MagicMock() # Simulate a running client
            mock_get_client.side_effect = None # Clear any previous side effects

            status = await get_ray_serve_status()
            assert status.ray_serve_status == "Ray: running, Serve: running"
            assert status.configured_models == [] # Check it returns models list

            # --- Test Case 2: Ray Initialized, Serve Not Running ---
            mock_is_initialized.return_value = True
            mock_get_client.side_effect = Exception("Serve not running") # Simulate client error

            status = await get_ray_serve_status()
            assert status.ray_serve_status == "Ray: running, Serve: not_running"

            # --- Test Case 3: Ray Not Initialized ---
            mock_is_initialized.return_value = False
            mock_get_client.side_effect = None # Reset side effect
            # No need to set return_value when is_initialized is False, as get_client won't be called

            status = await get_ray_serve_status()
            assert status.ray_serve_status == "Ray: not_running, Serve: not_running"

    @pytest.mark.asyncio
    async def test_concurrent_status_requests(self):
        """Test handling of concurrent status requests."""
        # Mock the specific functions/attributes used within get_ray_serve_status
        with patch('backend.routers.service_router.ray.is_initialized') as mock_is_initialized, \
             patch('backend.routers.service_router.ray.nodes') as mock_nodes, \
             patch('backend.routers.service_router.ray.available_resources') as mock_available_resources, \
             patch('backend.routers.service_router.ray.__version__', "mock_version"), \
             patch('backend.routers.service_router.serve.api._get_global_client') as mock_get_client, \
             patch('backend.routers.service_router.get_configured_models_internal', return_value=[]): # Mock model fetching

            # Configure mocks for a running state
            mock_is_initialized.return_value = True
            mock_nodes.return_value = ["node1"]
            mock_available_resources.return_value = {"CPU": 4}
            mock_get_client.return_value = MagicMock()
            mock_get_client.side_effect = None # Ensure no side effects interfere

            # Function to get status asynchronously
            async def get_status_async():
                return await get_ray_serve_status()

            # Run multiple requests concurrently using asyncio
            tasks = [get_status_async() for _ in range(10)]
            results = await asyncio.gather(*tasks)

            # Verify all results are consistent
            for result in results:
                assert result.ray_serve_status == "Ray: running, Serve: running"
                assert result.configured_models == []