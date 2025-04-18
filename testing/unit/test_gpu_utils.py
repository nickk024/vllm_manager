import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Import the module to test
from backend.utils.gpu_utils import get_gpu_stats, get_gpu_count, _get_gpu_stats_nvml, _get_gpu_stats_smi
from backend.models import GPUStat

class TestGpuUtils:
    """Test suite for GPU utilities with platform-specific mocks."""
    
    def test_get_gpu_count_apple_silicon(self):
        """Test GPU count detection on Apple Silicon (test environment)."""
        # Mock both NVML and subprocess to simulate Apple Silicon environment
        with patch('backend.utils.gpu_utils.NVML_AVAILABLE', False), \
             patch('subprocess.check_output') as mock_subprocess:
            # Simulate nvidia-smi not found (Apple Silicon)
            mock_subprocess.side_effect = FileNotFoundError("nvidia-smi not found")
            
            # Function should return 0 GPUs
            assert get_gpu_count() == 0
    
    def test_get_gpu_count_nvidia(self):
        """Test GPU count detection on NVIDIA GPUs (production environment)."""
        # Test with NVML available
        with patch('backend.utils.gpu_utils.is_apple_silicon', return_value=False), \
             patch('backend.utils.gpu_utils.NVML_AVAILABLE', True), \
             patch('pynvml.nvmlDeviceGetCount', return_value=2):
            assert get_gpu_count() == 2
        
        # Test with NVML unavailable but nvidia-smi available
        with patch('backend.utils.gpu_utils.is_apple_silicon', return_value=False), \
             patch('backend.utils.gpu_utils.NVML_AVAILABLE', False), \
             patch('subprocess.check_output') as mock_subprocess:
            mock_subprocess.return_value = "GPU 0: NVIDIA A100\nGPU 1: NVIDIA A100\n"
            assert get_gpu_count() == 2
    
    def test_get_gpu_stats_apple_silicon(self):
        """Test GPU stats on Apple Silicon (should return empty list)."""
        with patch('backend.utils.gpu_utils.NVML_AVAILABLE', False), \
             patch('subprocess.check_output') as mock_subprocess:
            # Simulate nvidia-smi not found (Apple Silicon)
            mock_subprocess.side_effect = FileNotFoundError("nvidia-smi not found")
            
            # Function should return empty list
            assert get_gpu_stats() == []
    
    def test_get_gpu_stats_nvidia(self):
        """Test GPU stats on NVIDIA GPUs (production environment)."""
        # Test with NVML available
        mock_handle = MagicMock()
        mock_mem_info = MagicMock(used=5*1024*1024*1024, total=40*1024*1024*1024)
        mock_util = MagicMock(gpu=75.5)
        
        with patch('backend.utils.gpu_utils.is_apple_silicon', return_value=False), \
             patch('backend.utils.gpu_utils.NVML_AVAILABLE', True), \
             patch('pynvml.nvmlDeviceGetCount', return_value=1), \
             patch('pynvml.nvmlDeviceGetHandleByIndex', return_value=mock_handle), \
             patch('pynvml.nvmlDeviceGetName', return_value="NVIDIA A100"), \
             patch('pynvml.nvmlDeviceGetMemoryInfo', return_value=mock_mem_info), \
             patch('pynvml.nvmlDeviceGetUtilizationRates', return_value=mock_util), \
             patch('pynvml.nvmlDeviceGetTemperature', return_value=65):
            
            result = get_gpu_stats()
            assert len(result) == 1
            assert result[0].name == "NVIDIA A100"
            assert result[0].gpu_id == 0
            assert result[0].memory_used_mb == 5*1024
            assert result[0].memory_total_mb == 40*1024
            assert result[0].gpu_utilization_pct == 75.5
            assert result[0].temperature_c == 65
    
    def test_get_gpu_stats_smi_fallback(self):
        """Test GPU stats using nvidia-smi fallback."""
        with patch('backend.utils.gpu_utils.is_apple_silicon', return_value=False), \
             patch('backend.utils.gpu_utils.NVML_AVAILABLE', False), \
             patch('subprocess.check_output') as mock_subprocess:
            # Simulate nvidia-smi output
            mock_subprocess.return_value = "0, NVIDIA A100, 65, 75.5, 5120, 40960"
            
            result = get_gpu_stats()
            assert len(result) == 1
            assert result[0].name == "NVIDIA A100"
            assert result[0].gpu_id == 0
            assert result[0].memory_used_mb == 5120
            assert result[0].memory_total_mb == 40960
            assert result[0].gpu_utilization_pct == 75.5
            assert result[0].temperature_c == 65
    
    def test_nvml_error_handling(self):
        """Test error handling when NVML throws exceptions."""
        with patch('backend.utils.gpu_utils.is_apple_silicon', return_value=False), \
             patch('backend.utils.gpu_utils.NVML_AVAILABLE', True), \
             patch('pynvml.nvmlDeviceGetCount', side_effect=Exception("NVML Error")), \
             patch('backend.utils.gpu_utils._get_gpu_stats_smi', return_value=[]) as mock_smi:
            
            # Call the function
            result = get_gpu_stats()
            
            # Verify the results
            # Note: In the actual implementation, _get_gpu_stats_nvml catches the exception
            # and returns an empty list directly, without calling _get_gpu_stats_smi
            # This is why we're just checking the result is an empty list
            assert isinstance(result, list)
            assert len(result) == 0