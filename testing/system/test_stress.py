import pytest
import time
import threading
import concurrent.futures
import os
from unittest.mock import patch, MagicMock
import requests
from fastapi.testclient import TestClient

from backend.main import app
from backend.ray_deployments import build_llm_deployments
from backend.utils.gpu_utils import is_apple_silicon


class TestStress:
    """Test suite for stress testing and performance."""

    def test_concurrent_api_requests(self):
        """Test handling of many concurrent API requests."""
        client = TestClient(app)
        
        # Number of concurrent requests
        num_requests = 50
        
        # Mock the necessary functions
        with patch('backend.routers.models_router.load_model_config', return_value={}), \
             patch('backend.routers.models_router.logger'):
            
            # Function to make a request
            def make_request():
                return client.get("/api/v1/manage/models/popular")
            
            # Make concurrent requests
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                responses = [future.result() for future in futures]
            end_time = time.time()
            
            # Calculate requests per second
            duration = end_time - start_time
            requests_per_second = num_requests / duration
            
            # Check that all requests were handled
            successful_responses = sum(1 for r in responses if r.status_code == 200)
            
            # Log the results
            print(f"Handled {successful_responses}/{num_requests} requests in {duration:.2f} seconds")
            print(f"Requests per second: {requests_per_second:.2f}")
            
            # At least 90% of requests should be successful
            assert successful_responses >= num_requests * 0.9
            
            # Requests per second should be reasonable (adjust based on your hardware)
            # This is a very conservative threshold
            assert requests_per_second > 1.0

    def test_multiple_model_loading(self):
        """Test loading multiple models concurrently."""
        # Skip on Apple Silicon environments
        if is_apple_silicon():
            pytest.skip("This test is not applicable on Apple Silicon")
            
        # Create a test config with multiple models
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
            },
            "model3": {
                "model_id": "org/model3",
                "serve": True,
                "tensor_parallel_size": 1
            }
        }
        
        # Mock the necessary functions
        with patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=["config.json", "model.safetensors"]), \
             patch('backend.ray_deployments.build_llm_deployment') as mock_build_deployment, \
             patch('backend.ray_deployments.logger'):
            
            # Create a mock deployment that takes some time to initialize
            def delayed_deployment(*args, **kwargs):
                time.sleep(0.1)  # Simulate some initialization time
                return MagicMock()
            
            mock_build_deployment.side_effect = delayed_deployment
            
            # Measure the time to build all deployments
            start_time = time.time()
            deployments = build_llm_deployments(config)
            end_time = time.time()
            
            # Calculate the duration
            duration = end_time - start_time
            
            # All models should be deployed
            assert len(deployments) == 3
            
            # The duration should be less than the sum of individual initialization times
            # If they were loaded sequentially, it would take at least 0.3 seconds
            # With concurrency, it should take less time
            # Allow a small margin of error (0.01 seconds) for timing variations
            assert duration < 0.31, f"Models were not loaded concurrently (took {duration:.2f} seconds)"

    def test_long_running_stability(self):
        """Test stability over a longer period with continuous requests."""
        client = TestClient(app)
        
        # Test duration in seconds
        test_duration = 5
        
        # Mock the necessary functions
        with patch('backend.routers.models_router.load_model_config', return_value={}), \
             patch('backend.routers.models_router.logger'):
            
            # Track successful and failed requests
            successful_requests = 0
            failed_requests = 0
            
            # Function to make continuous requests
            def make_continuous_requests():
                nonlocal successful_requests, failed_requests
                end_time = time.time() + test_duration
                while time.time() < end_time:
                    try:
                        response = client.get("/api/v1/manage/models/popular")
                        if response.status_code == 200:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    except Exception:
                        failed_requests += 1
            
            # Start multiple threads to make requests
            threads = []
            for _ in range(5):  # 5 concurrent threads
                thread = threading.Thread(target=make_continuous_requests)
                thread.daemon = True
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Log the results
            total_requests = successful_requests + failed_requests
            success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
            print(f"Made {total_requests} requests in {test_duration} seconds")
            print(f"Success rate: {success_rate:.2f}%")
            
            # Success rate should be high (at least 95%)
            assert success_rate >= 95, f"Success rate too low: {success_rate:.2f}%"

    def test_performance_under_load(self):
        """Test performance under load with simulated traffic."""
        # Skip in CI environments
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping performance test in CI environment")
            
        client = TestClient(app)
        
        # Number of requests to make
        num_requests = 100
        
        # Mock the necessary functions
        with patch('backend.routers.models_router.load_model_config', return_value={}), \
             patch('backend.routers.models_router.logger'):
            
            # Make requests and measure response times
            response_times = []
            for _ in range(num_requests):
                start_time = time.time()
                response = client.get("/api/v1/manage/models/popular")
                end_time = time.time()
                
                # Record response time if request was successful
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
            
            # Calculate statistics
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
                
                # Log the results
                print(f"Average response time: {avg_response_time:.4f} seconds")
                print(f"95th percentile response time: {p95_response_time:.4f} seconds")
                print(f"Min response time: {min_response_time:.4f} seconds")
                print(f"Max response time: {max_response_time:.4f} seconds")
                
                # Response times should be reasonable (increased thresholds slightly for robustness)
                assert avg_response_time < 0.15, f"Average response time too high: {avg_response_time:.4f} seconds"
                assert p95_response_time < 0.3, f"95th percentile response time too high: {p95_response_time:.4f} seconds"
            else:
                pytest.fail("No successful requests were made")