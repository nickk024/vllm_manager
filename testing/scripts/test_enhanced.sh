#!/bin/bash

# Enhanced test script for vllm_manager
# This script runs tests with additional features like stress tests and performance benchmarks

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}"; }

# Parse command line arguments
SKIP_STRESS=false
SKIP_BENCHMARK=false
SKIP_COVERAGE=false
VERBOSE=false

for arg in "$@"; do
    case $arg in
        --skip-stress)
            SKIP_STRESS=true
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --skip-coverage)
            SKIP_COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-stress     Skip stress tests"
            echo "  --skip-benchmark  Skip performance benchmarks"
            echo "  --skip-coverage   Skip coverage reporting"
            echo "  --verbose, -v     Enable verbose output"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
    esac
done

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
    pip install pytest-benchmark
else
    print_warning "Test virtual environment not found, creating one"
    python3 -m venv test_venv
    source test_venv/bin/activate
    print_info "Installing backend requirements"
    pip install -r backend/requirements.txt
    print_info "Installing test requirements"
    pip install -r testing/requirements-test.txt
    pip install pytest-benchmark
fi

# Run unit tests
print_step "Running Unit Tests"
if [ "$VERBOSE" = true ]; then
    python -m pytest testing/unit/ -v
else
    python -m pytest testing/unit/
fi

# Run integration tests
print_step "Running Integration Tests"
if [ "$VERBOSE" = true ]; then
    python -m pytest testing/integration/ -v
else
    python -m pytest testing/integration/
fi

# Run system tests
print_step "Running System Tests"
if [ "$VERBOSE" = true ]; then
    python -m pytest testing/system/ -v
else
    python -m pytest testing/system/
fi

# Run frontend tests
print_step "Running Frontend Tests"
if [ "$VERBOSE" = true ]; then
    python -m pytest testing/frontend/ -v
else
    python -m pytest testing/frontend/
fi

# Run stress tests if not skipped
if [ "$SKIP_STRESS" = false ]; then
    print_step "Running Stress Tests"
    if [ "$VERBOSE" = true ]; then
        python -m pytest testing/system/test_stress.py -v
    else
        python -m pytest testing/system/test_stress.py
    fi
else
    print_warning "Skipping stress tests (--skip-stress flag provided)"
fi

# Run performance benchmarks if not skipped
if [ "$SKIP_BENCHMARK" = false ]; then
    print_step "Running Performance Benchmarks"
    if [ "$VERBOSE" = true ]; then
        python -m pytest testing/system/ --benchmark-only -v
    else
        python -m pytest testing/system/ --benchmark-only
    fi
else
    print_warning "Skipping performance benchmarks (--skip-benchmark flag provided)"
fi

# Run coverage if not skipped
if [ "$SKIP_COVERAGE" = false ]; then
    print_step "Running Coverage Report"
    python -m pytest testing/ --cov=backend --cov=frontend --cov-report=term --cov-report=html
    print_info "Coverage report available in: $(pwd)/htmlcov/index.html"
else
    print_warning "Skipping coverage report (--skip-coverage flag provided)"
fi

# Print summary
print_step "Test Summary"
print_info "Environment: ${ENV_TYPE}"
print_info "Test Environment: ${TEST_ENV}"
print_info "Python Version: $(python --version)"
print_info "Packages:"
pip list | grep -E 'fastapi|pytest|ray|torch|vllm'

print_success "All tests completed"