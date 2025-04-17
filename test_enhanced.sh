#!/bin/bash
# Enhanced test script that handles both test (macOS/Apple Silicon) and production (Debian/NVIDIA) environments

# Get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Detect environment
if [[ "$(uname)" == "Darwin" ]]; then
    ENV_TYPE="macos"
    echo "=== Detected macOS test environment ==="
    # Check if Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "=== Running on Apple Silicon ==="
        export VLLM_TEST_ENV="apple_silicon"
    else
        echo "=== Running on Intel Mac ==="
        export VLLM_TEST_ENV="macos_intel"
    fi
else
    ENV_TYPE="linux"
    echo "=== Detected Linux environment ==="
    # Check for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        echo "=== NVIDIA GPUs detected ==="
        export VLLM_TEST_ENV="nvidia_gpu"
        # Get GPU count and models
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_MODELS=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
        echo "=== Found $GPU_COUNT GPUs: $GPU_MODELS ==="
    else
        echo "=== No NVIDIA GPUs detected, using CPU mode ==="
        export VLLM_TEST_ENV="linux_cpu"
    fi
fi

# Setup test environment
echo "=== Setting up test environment ==="

# Create virtual environment if it doesn't exist
if [ ! -d "${SCRIPT_DIR}/test_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/test_venv"
fi

# Activate virtual environment
source "${SCRIPT_DIR}/test_venv/bin/activate"

# Install test dependencies
echo "=== Installing dependencies ==="
pip install -r backend/requirements.txt
pip install pytest pytest-cov httpx

# Create test directories
mkdir -p "${SCRIPT_DIR}/test_logs"
mkdir -p "${SCRIPT_DIR}/test_config"
mkdir -p "${SCRIPT_DIR}/test_models"

# Set environment variables for testing
export VLLM_TEST_MODE=1
export VLLM_LOG_DIR="${SCRIPT_DIR}/test_logs"
export VLLM_CONFIG_DIR="${SCRIPT_DIR}/test_config"
export VLLM_MODELS_DIR="${SCRIPT_DIR}/test_models"

# Set environment variables for test detection
export TEST_ENV_TYPE=$ENV_TYPE
export TEST_ENV_SUBTYPE=$VLLM_TEST_ENV
if [[ "$VLLM_TEST_ENV" == "nvidia_gpu" ]]; then
    export TEST_GPU_COUNT=$GPU_COUNT
fi

# Run tests with coverage
echo "=== Running tests with coverage ==="
python -m pytest backend/tests/ -v --cov=backend --cov-report=term --cov-report=html:coverage_report

# Print test environment summary
echo "=== Test Environment Summary ==="
echo "Environment Type: $ENV_TYPE"
echo "Test Environment: $VLLM_TEST_ENV"
echo "Python Version: $(python --version)"
echo "Packages:"
pip list | grep -E 'pytest|fastapi|ray|vllm|torch|huggingface'

# Cleanup
echo "=== Test complete ==="
echo "Coverage report available in: ${SCRIPT_DIR}/coverage_report/index.html"
deactivate