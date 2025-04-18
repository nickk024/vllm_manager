#!/bin/bash

# run_all_tests.sh - Script to run all tests for vLLM Manager

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}"; }

# Detect environment
print_step "Detecting Environment"
if [[ "$(uname)" == "Darwin" ]]; then
    ENV_TYPE="macos"
    if [[ "$(uname -m)" == "arm64" ]]; then
        TEST_ENV="apple_silicon"
        print_info "Detected macOS on Apple Silicon"
    else
        TEST_ENV="macos_intel"
        print_info "Detected macOS on Intel"
    fi
else
    ENV_TYPE="linux"
    if command -v nvidia-smi &> /dev/null; then
        TEST_ENV="nvidia_gpu"
        print_info "Detected Linux with NVIDIA GPUs"
        nvidia-smi -L
    else
        TEST_ENV="linux_cpu"
        print_info "Detected Linux without NVIDIA GPUs"
    fi
fi

# Set environment variables for tests
export TEST_ENVIRONMENT="${TEST_ENV}"
export VLLM_TEST_MODE="true"

# Activate virtual environment if it exists
if [ -d "test_venv" ]; then
    print_info "Activating test virtual environment"
    source test_venv/bin/activate
else
    print_warning "Test virtual environment not found, creating one"
    python3 -m venv test_venv
    source test_venv/bin/activate
    pip install -r backend/requirements.txt
    pip install pytest pytest-cov httpx
fi

# Run basic tests first
print_step "Running Basic Tests"
python -m pytest backend/tests/test_config.py backend/tests/test_logging.py -v

# Run API tests
print_step "Running API Tests"
python -m pytest backend/tests/test_api.py -v

# Run utility tests
print_step "Running Utility Tests"
python -m pytest backend/tests/test_system_utils.py backend/tests/test_gpu_utils.py backend/tests/test_hf_utils.py -v

# Run deployment tests
print_step "Running Deployment Tests"
python -m pytest backend/tests/test_ray_deployments.py -v

# Run environment-specific tests
if [[ "$TEST_ENV" == "nvidia_gpu" ]]; then
    print_step "Running NVIDIA-specific Tests"
    python -m pytest backend/tests/test_nvidia_compat.py -v
else
    print_warning "Skipping NVIDIA-specific tests on non-NVIDIA environment"
fi

# Run error handling tests
print_step "Running Error Handling Tests"
python -m pytest backend/tests/test_download_error_handling.py -v

# Run concurrency tests
print_step "Running Concurrency Tests"
python -m pytest backend/tests/test_concurrency.py -v

# Run memory management tests
print_step "Running Memory Management Tests"
python -m pytest backend/tests/test_memory_management.py -v

# Run security tests
print_step "Running Security Tests"
python -m pytest backend/tests/test_security.py -v

# Run stress tests (only in production or with explicit flag)
if [[ "$TEST_ENV" == "nvidia_gpu" || "$1" == "--with-stress" ]]; then
    print_step "Running Stress Tests"
    python -m pytest backend/tests/test_stress.py -v
else
    print_warning "Skipping stress tests (only run in production or with --with-stress flag)"
fi

# Run all tests with coverage
print_step "Running All Tests with Coverage"
python -m pytest backend/tests/ --cov=backend --cov-report=term --cov-report=html -v

# Print summary
print_step "Test Summary"
print_info "Environment: ${ENV_TYPE}"
print_info "Test Environment: ${TEST_ENV}"
print_info "Python Version: $(python --version)"
print_info "Packages:"
pip list | grep -E 'fastapi|pytest|ray|torch|vllm'

print_success "All tests completed"
print_info "Coverage report available in: $(pwd)/htmlcov/index.html"