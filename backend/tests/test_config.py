import json
import os
import pytest
from unittest.mock import patch

# Adjust the import path based on how pytest discovers tests
# Assuming pytest runs from the project root or backend directory
try:
    from backend.config import load_model_config, save_model_config
except ImportError:
    # If running directly from tests dir, adjust path
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from backend.config import load_model_config, save_model_config

# Test Suite for Config File Handling
def test_load_model_config_file_not_found(tmp_path):
    """Test loading config when the file doesn't exist."""
    # Use tmp_path fixture provided by pytest for temporary directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "model_config.json"

    # Patch CONFIG_PATH to point to our temporary file
    with patch('backend.config.CONFIG_PATH', str(config_path)):
        config = load_model_config()
        assert config == {}

def test_load_model_config_empty_file(tmp_path):
    """Test loading config from an empty file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "model_config.json"
    config_path.touch() # Create empty file

    with patch('backend.config.CONFIG_PATH', str(config_path)):
        config = load_model_config()
        assert config == {}

def test_load_model_config_invalid_json(tmp_path):
    """Test loading config from a file with invalid JSON."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "model_config.json"
    config_path.write_text("this is not json")

    with patch('backend.config.CONFIG_PATH', str(config_path)):
        config = load_model_config()
        assert config == {} # Should return empty dict on decode error

def test_load_model_config_success(tmp_path):
    """Test loading a valid config file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "model_config.json"
    valid_data = {"model1": {"model_id": "id1", "serve": True}}
    config_path.write_text(json.dumps(valid_data))

    with patch('backend.config.CONFIG_PATH', str(config_path)):
        config = load_model_config()
        assert config == valid_data

def test_save_and_load_model_config(tmp_path):
    """Test saving config data and then loading it back."""
    config_dir = tmp_path / "config"
    # Note: save_model_config creates the directory if needed
    config_path = config_dir / "model_config.json"
    test_data = {
        "model_a": {"model_id": "org/model-a", "serve": False, "tensor_parallel_size": 1},
        "model_b": {"model_id": "org/model-b", "serve": True, "tensor_parallel_size": 2}
    }

    # Patch CONFIG_PATH for both save and load
    with patch('backend.config.CONFIG_PATH', str(config_path)):
        # Save the data
        save_model_config(test_data)

        # Verify file exists and content is correct
        assert config_path.exists()
        content = config_path.read_text()
        loaded_from_file = json.loads(content)
        assert loaded_from_file == test_data

        # Load it back using the function
        loaded_config = load_model_config()
        assert loaded_config == test_data

def test_save_model_config_creates_dir(tmp_path):
    """Test that save_model_config creates the directory if it doesn't exist."""
    config_dir = tmp_path / "non_existent_config_dir"
    config_path = config_dir / "model_config.json"

    assert not config_dir.exists() # Ensure dir doesn't exist initially

    test_data = {"model_c": {"model_id": "id_c"}}
    with patch('backend.config.CONFIG_PATH', str(config_path)):
        save_model_config(test_data)

    assert config_dir.exists() # Directory should now exist
    assert config_path.exists()
    loaded_from_file = json.loads(config_path.read_text())
    assert loaded_from_file == test_data