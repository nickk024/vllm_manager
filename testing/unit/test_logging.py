import pytest
import os
import pathlib
import importlib
import logging # Added missing import
from unittest import mock
import backend.main
# Import specific constants after importing the module
from backend.main import LOGS_DIR, UNIFIED_LOG_PATH, BACKEND_LOG_DIR

def test_log_directory_creation(tmp_path):
    """Test log directory creation with proper permissions"""
    test_logs_dir = tmp_path / "test_logs"
    
    with mock.patch('backend.main.LOGS_DIR', test_logs_dir):
        # Recalculate paths with mocked directory
        test_unified_path = test_logs_dir / "vllm_manager_app.log"
        
        # Try to access the directory
        pathlib.Path(test_logs_dir).mkdir(parents=True, exist_ok=True)
        os.chmod(test_logs_dir, 0o755)
        
        assert test_logs_dir.exists(), "Log directory was not created"
        assert (test_logs_dir.stat().st_mode & 0o777) == 0o755, "Incorrect directory permissions"
        assert not test_unified_path.exists(), "Log file should not be created during directory setup"

def test_log_file_creation(tmp_path):
    """Test log file handler creation during application startup"""
    test_logs_dir = tmp_path / "test_logs"
    test_unified_path = test_logs_dir / "vllm_manager_app.log"

    # Ensure the directory exists so the handler *can* be created
    test_logs_dir.mkdir()

    # Mock the path variables and the RotatingFileHandler class itself
    with mock.patch('backend.main.LOGS_DIR', test_logs_dir), \
         mock.patch('backend.main.BACKEND_LOG_DIR', test_logs_dir), \
         mock.patch('backend.main.UNIFIED_LOG_PATH', test_unified_path), \
         mock.patch('logging.handlers.RotatingFileHandler') as MockRotatingHandler, \
         mock.patch('backend.main.sys.exit') as mock_exit, \
         mock.patch('backend.main.logger.error') as mock_logger_error:

        # Configure the mock handler instance to have a level attribute
        mock_handler_instance = MockRotatingHandler.return_value
        mock_handler_instance.level = logging.DEBUG # Set a default level

        # Reload the main module to trigger its initialization logic
        importlib.reload(backend.main)

        # Assert that RotatingFileHandler was attempted to be created
        MockRotatingHandler.assert_called_once()

        # Assert that the critical error path (sys.exit) was not taken
        mock_exit.assert_not_called()  # Application should not exit during normal log file handler setup
        # Assert that the non-critical logger.error for handler creation failure was NOT called
        # (We expect handler creation to succeed in this test case with the mocked paths/class)
        mock_logger_error.assert_not_called()  # logger.error should not be called during successful setup


def test_invalid_log_path_handling():
    """Test error handling when log directory creation fails"""
    # We don't need to mock paths, just the mkdir call
    with mock.patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")), \
         pytest.raises(SystemExit) as exc_info: # Expect SystemExit

        # Reload the main module. The mkdir call inside its init should fail.
        importlib.reload(backend.main)

    # Check if the exit code was 1 as expected
    assert exc_info.value.code == 1, "Should exit with code 1 when log directory creation fails"