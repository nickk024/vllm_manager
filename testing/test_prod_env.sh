#!/bin/bash

# Production environment test script for vllm_manager
# This script is designed to run tests in the Debian/NVIDIA production environment

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Detect environment
echo -e "${GREEN}=== Detecting environment ===${NC}"
if [[ "$(uname)" == "Linux" ]]; then
    echo "Linux environment detected"
    ENV_TYPE="linux"
    
    # Check for NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}=== NVIDIA GPUs detected ===${NC}"
        ENV_SUBTYPE="nvidia_gpu"
        
        # Get GPU count and names
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        GPU_NAMES=$(nvidia-smi --list-gpus | awk -F': ' '{print $2}' | awk -F'(' '{print $1}' | tr '\n' ',' | sed 's/,$//')
        echo -e "${GREEN}=== Found ${GPU_COUNT} GPUs: ${GPU_NAMES} ===${NC}"
        
        # Check CUDA version
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
            echo -e "${GREEN}=== CUDA version: ${CUDA_VERSION} ===${NC}"
        else
            echo -e "${YELLOW}=== CUDA compiler not found, but continuing with NVIDIA drivers ===${NC}"
        fi
    else
        echo -e "${YELLOW}=== No NVIDIA GPUs detected, some tests will be skipped ===${NC}"
        ENV_SUBTYPE="linux_cpu"
    fi
else
    echo -e "${YELLOW}=== Non-Linux environment detected, some tests may fail ===${NC}"
    ENV_TYPE="other"
    ENV_SUBTYPE="unknown"
fi

# Setup test environment
echo -e "${GREEN}=== Setting up test environment ===${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "test_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv test_venv
fi

# Activate virtual environment
source test_venv/bin/activate

# Install dependencies
echo -e "${GREEN}=== Installing dependencies ===${NC}"
pip install -r backend/requirements.txt
pip install pytest pytest-cov httpx

# Run tests with coverage
echo -e "${GREEN}=== Running tests with coverage ===${NC}"

# Set environment variables for testing
export TEST_ENV_TYPE=$ENV_TYPE
export TEST_ENV_SUBTYPE=$ENV_SUBTYPE
export TEST_GPU_COUNT=$GPU_COUNT

# Run the tests
python -m pytest backend/tests/ -v --cov=backend --cov-report=term --cov-report=html:coverage_report

# Print environment summary
echo -e "${GREEN}=== Test Environment Summary ===${NC}"
echo "Environment Type: $ENV_TYPE"
echo "Test Environment: $ENV_SUBTYPE"
echo "Python Version: $(python --version)"
echo "Packages:"
pip list | grep -E "fastapi|huggingface-hub|prometheus-fastapi-instrumentator|pytest|ray|torch|vllm"

echo -e "${GREEN}=== Test complete ===${NC}"
echo "Coverage report available in: $(pwd)/coverage_report/index.html"

# Deactivate virtual environment
deactivate