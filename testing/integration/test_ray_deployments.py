import pytest
import os
from unittest.mock import patch, MagicMock

from backend.ray_deployments import build_llm_deployments

class TestRayDeployments:
    """Test suite for Ray Serve deployment builder."""
    
    def test_build_llm_deployments_empty_config(self):
        """Test building deployments with an empty config."""
        result = build_llm_deployments({})
        assert result is None
    
    def test_build_llm_deployments_no_serve_models(self):
        """Test building deployments when no models are marked to serve."""
        config = {
            "model1": {"model_id": "org/model1", "serve": False},
            "model2": {"model_id": "org/model2", "serve": False}
        }
        result = build_llm_deployments(config)
        assert result is None
    
    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('ray.serve.llm.build_llm_deployment')
    def test_build_llm_deployments_success(self, mock_build_deployment, mock_listdir, mock_isdir):
        """Test successful building of deployments."""
        # Mock directory checks to simulate downloaded models
        mock_isdir.return_value = True
        mock_listdir.return_value = ["config.json", "model.safetensors"]

        # Mock the build_llm_deployment function to return a mock deployment
        # and accept the arguments the code tries to pass, including positional path.
        mock_deployment = MagicMock(name="MockDeployment")
        def mock_builder_side_effect(*args, model_id=None, model_path=None, path=None, engine_config=None, **kwargs):
            # Check if path is provided positionally
            pos_path = args[0] if args else None
            # Basic check to ensure engine_config is passed and is a dict
            if engine_config is not None and not isinstance(engine_config, dict):
                 raise TypeError("engine_config must be a dictionary if provided")
            # Check that one of the path arguments is provided (keyword or positional)
            if not (model_id or model_path or path or pos_path):
                 raise TypeError("A model path argument (model_id, model_path, path, or positional) is required")
            # Simulate the actual function's behavior based on which args are present
            # This mock just needs to accept the call signature and return the mock deployment
            return mock_deployment
        mock_build_deployment.side_effect = mock_builder_side_effect

        # Create test config with models marked to serve
        config = {
            "model1": {
                "model_id": "org/model1",
                "serve": True,
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
                "dtype": "bfloat16"
            },
            "model2": {
                "model_id": "org/model2",
                "serve": True,
                "tensor_parallel_size": 2,
                "max_model_len": 8192,
                "dtype": "float16"
            },
            "model3": {
                "model_id": "org/model3",
                "serve": False  # This one should be skipped
            }
        }

        # Call the function
        result = build_llm_deployments(config) # Call the original function directly

        # Verify the results
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 2 # Only model1 and model2 should be included
        assert "model1" in result
        assert "model2" in result
        assert result["model1"] is mock_deployment
        assert result["model2"] is mock_deployment

        # Verify mock calls
        assert mock_build_deployment.call_count == 2
        # Check call args for model1
        call_args_model1 = mock_build_deployment.call_args_list[0][1] # kwargs of first call
        assert call_args_model1['model_id'] == "org/model1"
        assert call_args_model1['tensor_parallel_size'] == 1
        assert call_args_model1['max_model_len'] == 4096
        assert call_args_model1['dtype'] == "bfloat16"
        # Check call args for model2
        call_args_model2 = mock_build_deployment.call_args_list[1][1] # kwargs of second call
        assert call_args_model2['model_id'] == "org/model2"
        assert call_args_model2['tensor_parallel_size'] == 2
        assert call_args_model2['max_model_len'] == 8192
        assert call_args_model2['dtype'] == "float16"
    
    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('ray.serve.llm.build_llm_deployment')
    def test_build_llm_deployments_not_downloaded(self, mock_build_deployment, mock_listdir, mock_isdir):
        """Test when models are marked to serve but not downloaded."""
        # Mock directory checks to simulate models not downloaded
        mock_isdir.return_value = True
        mock_listdir.return_value = []  # Empty directory
        
        # Create test config with models marked to serve
        config = {
            "model1": {"model_id": "org/model1", "serve": True},
            "model2": {"model_id": "org/model2", "serve": True}
        }
        
        # Call the function
        result = build_llm_deployments(config)
        
        # Verify the results
        assert result is None
        assert mock_build_deployment.call_count == 0
    
    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('ray.serve.llm.build_llm_deployment')
    def test_build_llm_deployments_partial_success(self, mock_build_deployment, mock_listdir, mock_isdir):
        """Test when some deployments succeed and others fail."""
        # Mock directory checks to simulate downloaded models
        mock_isdir.return_value = True
        mock_listdir.return_value = ["config.json", "model.safetensors"]

        # Mock the build_llm_deployment function to succeed for first model and fail for second
        mock_deployment_success = MagicMock(name="SuccessDeployment")

        # Define a side_effect function that accepts potential args and simulates partial failure
        call_count = [0]
        def mock_builder_side_effect(*args, model_id=None, model_path=None, path=None, engine_config=None, **kwargs):
            call_count[0] += 1
            # Check if path is provided positionally
            pos_path = args[0] if args else None
            # Basic check to ensure engine_config is passed and is a dict
            if engine_config is not None and not isinstance(engine_config, dict):
                 raise TypeError("engine_config must be a dictionary if provided")
            # Check that one of the path arguments is provided (keyword or positional)
            if not (model_id or model_path or path or pos_path):
                 raise TypeError("A model path argument (model_id, model_path, path, or positional) is required")

            if call_count[0] == 1:  # First call (model1)
                return mock_deployment_success
            else:  # Second call (model2)
                # Raise the specific error the code expects to catch if needed,
                # or a generic one for testing the overall failure handling.
                raise Exception("Simulated deployment build error for model2")

        mock_build_deployment.side_effect = mock_builder_side_effect

        # Create test config with models marked to serve
        config = {
            "model1": {"model_id": "org/model1", "serve": True, "tensor_parallel_size": 1},
            "model2": {"model_id": "org/model2", "serve": True, "tensor_parallel_size": 1}
        }

        # Call the function
        result = build_llm_deployments(config) # Call original function

        # Verify the results - should contain only the successful deployment
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 1
        assert "model1" in result
        assert "model2" not in result # model2 failed
        assert result["model1"] is mock_deployment_success

        # Verify mock calls
        assert mock_build_deployment.call_count == 2
        # Check call args for model1 (successful)
        call_args_model1 = mock_build_deployment.call_args_list[0][1]
        assert call_args_model1['model_id'] == "org/model1"
        # Check call args for model2 (failed)
        call_args_model2 = mock_build_deployment.call_args_list[1][1]
        assert call_args_model2['model_id'] == "org/model2"
    
    @patch('os.path.isdir')
    @patch('os.listdir')
    @patch('ray.serve.llm.build_llm_deployment')
    def test_build_llm_deployments_invalid_config(self, mock_build_deployment, mock_listdir, mock_isdir):
        """Test handling of invalid config entries."""
        # Mock directory checks
        mock_isdir.return_value = True
        mock_listdir.return_value = ["config.json"]

        # Mock the build_llm_deployment function to accept potential args
        mock_deployment_valid = MagicMock(name="ValidDeployment")
        def mock_builder_side_effect(*args, model_id=None, model_path=None, path=None, engine_config=None, **kwargs):
            # Check if path is provided positionally
            pos_path = args[0] if args else None
             # Basic check to ensure engine_config is passed and is a dict
            if engine_config is not None and not isinstance(engine_config, dict):
                 raise TypeError("engine_config must be a dictionary if provided")
            # Check that one of the path arguments is provided (keyword or positional)
            if not (model_id or model_path or path or pos_path):
                 raise TypeError("A model path argument (model_id, model_path, path, or positional) is required")
            return mock_deployment_valid
        mock_build_deployment.side_effect = mock_builder_side_effect

        # Config with invalid entries
        config = {
            "model1": {"model_id": "org/model1", "serve": True, "tensor_parallel_size": 1},  # Valid
            "model2": "not_a_dict",  # Invalid
            "model3": None,  # Invalid
            "model4": {"model_id": "org/model4", "serve": False}, # Valid but not served
            "model5": {"model_id": "org/model5", "serve": True, "tensor_parallel_size": 1} # Valid
        }

        # Call the function
        result = build_llm_deployments(config)

        # Verify the results - should contain only valid, served deployments
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "model1" in result
        assert "model5" in result
        assert "model2" not in result
        assert "model3" not in result
        assert "model4" not in result
        assert result["model1"] is mock_deployment_valid
        assert result["model5"] is mock_deployment_valid

        # Verify mock calls - should only be called for model1 and model5
        assert mock_build_deployment.call_count == 2
        call_args_model1 = mock_build_deployment.call_args_list[0][1]
        assert call_args_model1['model_id'] == "org/model1"
        call_args_model5 = mock_build_deployment.call_args_list[1][1]
        assert call_args_model5['model_id'] == "org/model5"