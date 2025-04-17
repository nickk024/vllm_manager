import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import platform

from backend.utils.gpu_utils import get_gpu_count, get_gpu_stats
from backend.ray_deployments import build_llm_deployments


def is_nvidia_environment():
    """Check if the current environment has NVIDIA GPUs."""
    # Skip on macOS
    if platform.system() == "Darwin":
        return False
    
    # Try to import pynvml
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
            return True
        except:
            return False
    except ImportError:
        return False


class TestNvidiaCompat:
    """Test suite for NVIDIA GPU compatibility in production environment."""

    def test_nvidia_gpu_count(self):
        """Test GPU count detection with NVIDIA GPUs."""
        # Skip on non-NVIDIA environments
        if not is_nvidia_environment():
            pytest.skip("This test requires an NVIDIA environment")
            
        with patch('backend.utils.gpu_utils.is_apple_silicon', return_value=False), \
             patch('pynvml.nvmlInit') as mock_init, \
             patch('pynvml.nvmlDeviceGetCount') as mock_get_count:
            
            # Mock NVML functions
            mock_init.return_value = None
            mock_get_count.return_value = 4  # Simulate 4 NVIDIA GPUs
            
            # Get GPU count
            count = get_gpu_count()
            
            # Verify the result
            assert count == 4
            mock_init.assert_called_once()
            mock_get_count.assert_called_once()

    def test_nvidia_gpu_stats(self):
        """Test GPU stats collection with NVIDIA GPUs."""
        # Skip on non-NVIDIA environments
        if not is_nvidia_environment():
            pytest.skip("This test requires an NVIDIA environment")
            
        with patch('backend.utils.gpu_utils.is_apple_silicon', return_value=False), \
             patch('pynvml.nvmlInit') as mock_init, \
             patch('pynvml.nvmlDeviceGetCount') as mock_count, \
             patch('pynvml.nvmlDeviceGetHandleByIndex') as mock_handle, \
             patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_mem, \
             patch('pynvml.nvmlDeviceGetUtilizationRates') as mock_util, \
             patch('pynvml.nvmlDeviceGetTemperature') as mock_temp:
            
            # Mock NVML functions
            mock_init.return_value = None
            mock_count.return_value = 2  # Simulate 2 NVIDIA GPUs
            
            # Create mock handles
            handle1 = MagicMock()
            handle2 = MagicMock()
            mock_handle.side_effect = [handle1, handle2]
            
            # Mock memory info
            mem1 = MagicMock()
            mem1.used = 5000000000  # 5 GB used
            mem1.total = 24000000000  # 24 GB total
            mem2 = MagicMock()
            mem2.used = 8000000000  # 8 GB used
            mem2.total = 24000000000  # 24 GB total
            mock_mem.side_effect = [mem1, mem2]
            
            # Mock utilization rates
            util1 = MagicMock()
            util1.gpu = 30  # 30% GPU utilization
            util1.memory = 20  # 20% memory utilization
            util2 = MagicMock()
            util2.gpu = 75  # 75% GPU utilization
            util2.memory = 60  # 60% memory utilization
            mock_util.side_effect = [util1, util2]
            
            # Mock temperatures
            mock_temp.side_effect = [50, 65]  # 50°C and 65°C
            
            # Get GPU stats
            stats = get_gpu_stats()
            
            # Verify the results
            assert len(stats) == 2
            
            assert stats[0]["index"] == 0
            assert stats[0]["memory_used_gb"] == 5.0
            assert stats[0]["memory_total_gb"] == 24.0
            assert stats[0]["utilization_gpu"] == 30
            assert stats[0]["utilization_memory"] == 20
            assert stats[0]["temperature"] == 50
            
            assert stats[1]["index"] == 1
            assert stats[1]["memory_used_gb"] == 8.0
            assert stats[1]["memory_total_gb"] == 24.0
            assert stats[1]["utilization_gpu"] == 75
            assert stats[1]["utilization_memory"] == 60
            assert stats[1]["temperature"] == 65

    def test_tensor_parallel_with_multiple_gpus(self):
        """Test tensor parallel configuration with multiple GPUs."""
        # Skip on non-NVIDIA environments
        if not is_nvidia_environment():
            pytest.skip("This test requires an NVIDIA environment")
            
        with patch('os.path.isdir') as mock_isdir, \
             patch('os.listdir') as mock_listdir, \
             patch('ray.serve.llm.build_llm_deployment') as mock_build_deployment:
            
            # Mock directory checks to simulate downloaded models
            mock_isdir.return_value = True
            mock_listdir.return_value = ["config.json", "model.safetensors"]
            
            # Mock the build_llm_deployment function
            mock_deployment = MagicMock()
            mock_build_deployment.return_value = mock_deployment
            
            # Create test config with tensor_parallel_size > 1
            config = {
                "large_model": {
                    "model_id": "org/large-model",
                    "serve": True,
                    "tensor_parallel_size": 4,  # Use 4 GPUs
                    "max_model_len": 8192,
                    "dtype": "bfloat16"
                }
            }
            
            # Call the function
            with patch('backend.ray_deployments.logger') as mock_logger:
                result = build_llm_deployments(config)
                
                # Verify that the function attempted to build with tensor_parallel_size=4
                assert mock_logger.info.called
                # Check if any log message contains tensor_parallel_size
                tp_size_logged = False
                for call in mock_logger.info.call_args_list:
                    args, _ = call
                    if args and "tensor_parallel_size" in str(args[0]):
                        tp_size_logged = True
                        break
                assert tp_size_logged

    def test_debian_environment_variables(self):
        """Test environment variables for Debian compatibility."""
        # Skip this test if not running on Linux
        if not sys.platform.startswith('linux'):
            pytest.skip("This test is only relevant on Linux systems")
            
        # Skip on non-NVIDIA environments
        if not is_nvidia_environment():
            pytest.skip("This test requires an NVIDIA environment")
            
        # Check for CUDA-related environment variables
        cuda_home = os.environ.get('CUDA_HOME')
        cuda_path = os.environ.get('CUDA_PATH')
        ld_library_path = os.environ.get('LD_LIBRARY_PATH')
        
        # Log the environment variables for debugging
        print(f"CUDA_HOME: {cuda_home}")
        print(f"CUDA_PATH: {cuda_path}")
        print(f"LD_LIBRARY_PATH: {ld_library_path}")
        
        # This test is informational and doesn't assert anything
        # It helps identify potential issues in the production environment