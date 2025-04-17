import pytest
from unittest.mock import patch, MagicMock
import subprocess

from backend.utils.system_utils import get_system_stats
from backend.models import SystemStats

class TestSystemUtils:
    """Test suite for system utilities with platform-specific mocks."""
    
    def test_get_system_stats_success(self):
        """Test successful system stats collection."""
        # Mock both subprocess calls
        with patch('subprocess.check_output') as mock_subprocess:
            # Set up the mock to return different values for different commands
            def side_effect(cmd, **kwargs):
                if 'top -bn1' in cmd:
                    return "25.5"
                elif 'free -m' in cmd:
                    return "Mem:        16384       4096      12288        128       1024       8192"
                return ""
            
            mock_subprocess.side_effect = side_effect
            
            # Get the stats
            result = get_system_stats()
            
            # Verify the results
            assert isinstance(result, SystemStats)
            assert result.cpu_utilization_pct == 25.5
            assert result.memory_total_mb == 16384
            assert result.memory_used_mb == 4096
            assert result.memory_utilization_pct == 25.0
    
    def test_get_system_stats_command_not_found(self):
        """Test handling when commands are not found."""
        with patch('subprocess.check_output') as mock_subprocess:
            # Simulate command not found
            mock_subprocess.side_effect = FileNotFoundError("Command not found")
            
            # Get the stats
            result = get_system_stats()
            
            # Verify the results - should return an empty SystemStats object
            assert isinstance(result, SystemStats)
            assert result.cpu_utilization_pct is None
            assert result.memory_total_mb is None
            assert result.memory_used_mb is None
            assert result.memory_utilization_pct is None
    
    def test_get_system_stats_command_error(self):
        """Test handling when commands return errors."""
        with patch('subprocess.check_output') as mock_subprocess:
            # Simulate command error
            mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cmd", "error output")
            
            # Get the stats
            result = get_system_stats()
            
            # Verify the results - should return an empty SystemStats object
            assert isinstance(result, SystemStats)
            assert result.cpu_utilization_pct is None
            assert result.memory_total_mb is None
            assert result.memory_used_mb is None
            assert result.memory_utilization_pct is None
    
    def test_get_system_stats_parse_error(self):
        """Test handling when command output cannot be parsed."""
        with patch('subprocess.check_output') as mock_subprocess:
            # Set up the mock to return invalid values
            def side_effect(cmd, **kwargs):
                if 'top -bn1' in cmd:
                    return "not a number"
                elif 'free -m' in cmd:
                    return "Invalid format"
                return ""
            
            mock_subprocess.side_effect = side_effect
            
            # Get the stats
            result = get_system_stats()
            
            # Verify the results - should return an empty SystemStats object
            assert isinstance(result, SystemStats)
            assert result.cpu_utilization_pct is None
            assert result.memory_total_mb is None
            assert result.memory_used_mb is None
            assert result.memory_utilization_pct is None
    
    def test_get_system_stats_partial_success(self):
        """Test when only some stats can be collected."""
        with patch('subprocess.check_output') as mock_subprocess:
            # Set up the mock to succeed for CPU but fail for memory
            def side_effect(cmd, **kwargs):
                if 'top -bn1' in cmd:
                    return "25.5"
                elif 'free -m' in cmd:
                    raise subprocess.CalledProcessError(1, "cmd", "error output")
                return ""
            
            mock_subprocess.side_effect = side_effect
            
            # Get the stats
            result = get_system_stats()
            
            # Verify the results - should have CPU but not memory
            assert isinstance(result, SystemStats)
            assert result.cpu_utilization_pct == 25.5
            assert result.memory_total_mb is None
            assert result.memory_used_mb is None
            assert result.memory_utilization_pct is None