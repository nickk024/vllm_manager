import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

# Ensure backend is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Detect test environment
TEST_ENV = os.environ.get('VLLM_TEST_ENV', 'unknown')
IS_APPLE_SILICON = TEST_ENV == 'apple_silicon'
IS_NVIDIA_GPU = TEST_ENV == 'nvidia_gpu'

@pytest.fixture(scope="session")
def test_environment():
    """Return information about the test environment."""
    return {
        "env_type": TEST_ENV,
        "has_nvidia_gpus": IS_NVIDIA_GPU,
        "is_apple_silicon": IS_APPLE_SILICON,
        "python_version": sys.version,
        "os_name": os.name,
        "platform": sys.platform
    }

@pytest.fixture(scope="function")
def temp_config_dir():
    """Create a temporary directory for config files."""
    temp_dir = tempfile.mkdtemp(prefix="vllm_test_config_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def temp_models_dir():
    """Create a temporary directory for model files."""
    temp_dir = tempfile.mkdtemp(prefix="vllm_test_models_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def temp_logs_dir():
    """Create a temporary directory for log files."""
    temp_dir = tempfile.mkdtemp(prefix="vllm_test_logs_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def mock_config_paths(temp_config_dir, temp_models_dir, temp_logs_dir):
    """Mock the config paths to use temporary directories."""
    with patch('backend.config.CONFIG_DIR', temp_config_dir), \
         patch('backend.config.MODELS_DIR', temp_models_dir), \
         patch('backend.config.LOGS_DIR', temp_logs_dir), \
         patch('backend.config.CONFIG_PATH', os.path.join(temp_config_dir, "model_config.json")):
        yield {
            "config_dir": temp_config_dir,
            "models_dir": temp_models_dir,
            "logs_dir": temp_logs_dir,
            "config_path": os.path.join(temp_config_dir, "model_config.json")
        }

@pytest.fixture(scope="function")
def mock_gpu_environment():
    """Mock GPU environment based on test platform."""
    if IS_NVIDIA_GPU:
        # On NVIDIA systems, use real GPU utilities
        yield None
    else:
        # On non-NVIDIA systems, mock GPU utilities
        with patch('backend.utils.gpu_utils.NVML_AVAILABLE', False), \
             patch('backend.utils.gpu_utils._get_gpu_stats_smi', return_value=[]), \
             patch('backend.utils.gpu_utils.get_gpu_count', return_value=0):
            yield None

@pytest.fixture(scope="function")
def mock_ray_environment():
    """Mock Ray environment for testing."""
    # Create a more comprehensive Ray mock to avoid core_worker attribute errors
    mock_worker = MagicMock()
    mock_worker.worker_id.hex.return_value = "test_worker_id"
    
    with patch('ray.init'), \
         patch('ray.is_initialized', return_value=True), \
         patch('ray.serve.run'), \
         patch('ray.serve.shutdown'), \
         patch('ray._private.worker.Worker.core_worker', create=True), \
         patch('ray.runtime_context.get_runtime_context') as mock_context, \
         patch('ray._private.runtime_context.RuntimeContext.worker', mock_worker, create=True):
        
        # Configure the mock context
        mock_context.return_value.worker = mock_worker
        
        yield None

@pytest.fixture(scope="function")
def mock_vllm_environment():
    """Mock vLLM environment for testing."""
    with patch('backend.ray_deployments.VLLM_AVAILABLE', True), \
         patch('backend.ray_deployments.AsyncEngineArgs'), \
         patch('backend.ray_deployments.AsyncLLMEngine'):
        yield None

@pytest.fixture(scope="function")
def fastapi_test_client():
    """Create a FastAPI test client with mocked dependencies."""
    from fastapi.testclient import TestClient
    
    # Create a mock worker for Ray
    mock_worker = MagicMock()
    mock_worker.worker_id.hex.return_value = "test_worker_id"
    
    # Import the app with mocked dependencies
    with patch('backend.main.ray.init'), \
         patch('backend.main.ray.is_initialized', return_value=True), \
         patch('backend.main.serve.run'), \
         patch('backend.main.serve.shutdown'), \
         patch('backend.ray_deployments.build_llm_deployments', return_value={}), \
         patch('ray._private.worker.Worker.core_worker', create=True), \
         patch('ray.runtime_context.get_runtime_context') as mock_context, \
         patch('ray._private.runtime_context.RuntimeContext.worker', mock_worker, create=True):
        
        # Configure the mock context
        mock_context.return_value.worker = mock_worker
        
        # Disable Ray logging filters to avoid worker_id issues
        with patch('ray._private.ray_logging.filters.LoggingFilter.filter', return_value=True):
            from backend.main import app
            
            # Create and return the test client
            with TestClient(app) as client:
                yield client