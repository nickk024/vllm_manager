#!/bin/bash
# Test script that runs in isolated environment

# Get script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Setup clean venv
echo "=== Setting up test environment ==="
python3 -m venv "${SCRIPT_DIR}/test_venv"
source "${SCRIPT_DIR}/test_venv/bin/activate"

# Install test dependencies
pip install -r backend/requirements.txt
pip install pytest httpx

# Run tests with clean config
echo "=== Running tests ==="
export VLLM_TEST_MODE=1
export VLLM_LOG_DIR="${SCRIPT_DIR}/test_logs"
python -m pytest backend/tests/ -v

# Cleanup
deactivate
echo "=== Test complete ==="