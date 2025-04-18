import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json
from fastapi import HTTPException

from backend.routers.download_router import _download_single_model, run_download_task
from backend.models import ModelInfo


class TestDownloadErrorHandling:
    """Test suite for error handling in model download functionality."""

    def test_download_model_network_error(self):
        """Test handling of network errors during model download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the huggingface_hub.snapshot_download function to raise a network error
            with patch('backend.routers.download_router.snapshot_download') as mock_download, \
                 patch('backend.routers.download_router.logger') as mock_logger, \
                 patch.dict(os.environ, {"MODELS_DIR": temp_dir}):
                
                # Simulate a network error
                mock_download.side_effect = Exception("Network error")
                
                # Call the function with a test model config
                model_config = {
                    "model_id": "test/model",
                    "serve": True,
                    "tensor_parallel_size": 1
                }
                
                # Verify that the function handles the error gracefully
                result = _download_single_model("test_model", model_config, temp_dir)
                
                # Check that the error is properly logged
                assert mock_logger.error.called
                assert "Error downloading" in str(mock_logger.error.call_args)
                
                # Check that the function returns False on failure
                assert result is False

    @patch('backend.routers.download_router.snapshot_download')
    @patch('backend.routers.download_router.logger')
    @patch('os.makedirs') # Mock os.makedirs used in the function
    def test_download_model_permission_error(self, mock_os_makedirs, mock_logger, mock_snapshot_download):
        """Test handling of permission errors during model download (simulated)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate PermissionError when trying to create the model directory
            mock_os_makedirs.side_effect = PermissionError("Cannot create directory")

            # Call the function with a test model config
            model_config = {"model_id": "test/permission-model"}

            # Assert that PermissionError is raised directly from os.makedirs
            with pytest.raises(PermissionError, match="Cannot create directory"):
                _download_single_model("perm_model", model_config, temp_dir)

            # Verify that logger.error was NOT called within _download_single_model
            # because the exception happens before the try-except block
            mock_logger.error.assert_not_called()
            # snapshot_download should not be called if directory creation fails
            mock_snapshot_download.assert_not_called()

    @patch('backend.routers.download_router._download_single_model')
    @patch('backend.routers.download_router.logger')
    def test_download_all_models_partial_failure(self, mock_logger, mock_download_single):
        """Test handling of partial failures when downloading multiple models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_to_download = {
                "model_ok": {"model_id": "org/ok"},
                "model_fail": {"model_id": "org/fail"},
                "model_ok_too": {"model_id": "org/ok_too"}
            }
            download_dir = os.path.join(temp_dir, "models")
            token = "fake_token"
            force = False

            # Mock _download_single_model to succeed for some, fail for others
            # Correct signature: model_name, model_config, output_dir, token, force
            def download_side_effect(model_name, config, output_dir, token, force):
                if model_name == "model_fail":
                    return False # Simulate failure
                else:
                    return True # Simulate success

            mock_download_single.side_effect = download_side_effect

            # Run the task (note: run_download_task runs in a separate thread usually,
            # but here we call it directly for testing the logic within the same thread)
            # We expect it to log errors but not raise exceptions itself.
            run_download_task(models_to_download, download_dir, token, force)

            # Verify _download_single_model was called for all models
            assert mock_download_single.call_count == len(models_to_download)

            # Verify logger calls for start, success, failure, and completion
            # Check for specific log messages indicating start, success, failure, and completion
            log_calls = [call[0][0] for call in mock_logger.info.call_args_list] # Get positional args of calls
            # Correct the expected log message based on the actual code
            assert any("Background download task started for 3 models" in msg for msg in log_calls)
            assert any("Successfully downloaded model_ok" in msg for msg in log_calls)
            assert any("Successfully downloaded model_ok_too" in msg for msg in log_calls)
            # Failure is logged as error
            mock_logger.error.assert_called_once()
            assert "Failed to download model_fail" in mock_logger.error.call_args[0][0]
            assert any("Download task finished. Success: 2, Failed: 1" in msg for msg in log_calls)