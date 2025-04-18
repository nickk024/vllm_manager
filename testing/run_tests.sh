#!/bin/bash

# run_tests.sh - Main script to run all tests for vLLM Manager

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
    # Always ensure test requirements are installed
    print_info "Installing/updating test requirements"
    pip install -r testing/requirements-test.txt
else
    print_warning "Test virtual environment not found, creating one"
    python3 -m venv test_venv
    source test_venv/bin/activate
    print_info "Installing backend requirements"
    pip install -r backend/requirements.txt
    print_info "Installing test requirements"
    pip install -r testing/requirements-test.txt
fi

# Run all tests with coverage
# This single command runs unit, integration, system, and frontend tests
# It automatically handles environment-specific tests based on markers/fixtures
# Stress tests are typically marked and can be included/excluded via pytest arguments if needed
print_step "Running All Tests with Coverage"
# Add -v for verbose output, adjust cov targets as needed
python -m pytest testing/ --cov=backend --cov=frontend --cov-report=term --cov-report=html -v

# Print summary
print_step "Test Summary"
print_info "Environment: ${ENV_TYPE}"
print_info "Test Environment: ${TEST_ENV}"
print_info "Python Version: $(python --version)"
print_info "Packages:"
pip list | grep -E 'fastapi|pytest|ray|torch|vllm'

print_success "All tests completed"
print_info "Coverage report available in: $(pwd)/htmlcov/index.html"