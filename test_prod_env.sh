#!/bin/bash

# Test script for production environment with NVIDIA GPUs
# This script is designed to run tests in a Debian environment with NVIDIA GPUs

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Production Environment Test Script ===${NC}"

# Detect environment
if [ -f /etc/debian_version ]; then
    echo -e "${GREEN}=== Detected Debian-based system ===${NC}"
else
    echo -e "${YELLOW}=== Warning: Not running on Debian. Some tests may not be relevant. ===${NC}"
fi

# Check for NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}=== NVIDIA GPUs detected ===${NC}"
    nvidia-smi
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader)
    echo -e "${GREEN}=== Found $GPU_COUNT NVIDIA GPUs ===${NC}"
else
    echo -e "${RED}=== No NVIDIA GPUs detected. This script is intended for NVIDIA environments. ===${NC}"
    echo -e "${YELLOW}=== Continuing with limited testing... ===${NC}"
fi

# Check CUDA installation
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}=== CUDA installation detected ===${NC}"
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}=== CUDA version: $CUDA_VERSION ===${NC}"
else
    echo -e "${YELLOW}=== CUDA toolkit not found in PATH. Some tests may fail. ===${NC}"
fi

# Set up environment
echo -e "${YELLOW}=== Setting up test environment ===${NC}"

# Activate virtual environment if it exists, otherwise create it
if [ -d "venv" ]; then
    echo -e "${GREEN}=== Using existing virtual environment ===${NC}"
    source venv/bin/activate
else
    echo -e "${GREEN}=== Creating new virtual environment ===${NC}"
    python3 -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo -e "${YELLOW}=== Installing dependencies ===${NC}"
pip install -r backend/requirements.txt
pip install pytest pytest-cov httpx

# Run specific NVIDIA compatibility tests
echo -e "${YELLOW}=== Running NVIDIA compatibility tests ===${NC}"
python -m pytest backend/tests/test_nvidia_compat.py -v

# Run all tests with coverage
echo -e "${YELLOW}=== Running all tests with coverage ===${NC}"
python -m pytest backend/tests/ --cov=backend --cov-report=html:coverage_report_prod

# Test tensor parallel capabilities if multiple GPUs are available
if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 1 ]; then
    echo -e "${YELLOW}=== Testing tensor parallel capabilities with $GPU_COUNT GPUs ===${NC}"
    # Create a test script to verify tensor parallel functionality
    cat > test_tensor_parallel.py << EOF
import ray
from ray import serve
from vllm import LLM, SamplingParams
import time

# Initialize Ray and Serve
ray.init()
serve.start()

# Test with a small model using tensor parallelism
model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Change to an appropriate model
tp_size = min($GPU_COUNT, 2)  # Use at most 2 GPUs for the test

print(f"Testing tensor parallelism with {tp_size} GPUs")
start_time = time.time()

try:
    # Initialize the model with tensor parallelism
    llm = LLM(model=model_id, tensor_parallel_size=tp_size)
    
    # Simple inference test
    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
    prompts = ["Explain how tensor parallelism works in large language models."]
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated text: {output.outputs[0].text}")
        
    print(f"Test completed successfully in {time.time() - start_time:.2f} seconds")
    success = True
except Exception as e:
    print(f"Test failed: {e}")
    success = False

# Shutdown Ray
ray.shutdown()

# Exit with appropriate status code
exit(0 if success else 1)
EOF

    # Run the tensor parallel test
    python test_tensor_parallel.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}=== Tensor parallel test passed ===${NC}"
    else
        echo -e "${RED}=== Tensor parallel test failed ===${NC}"
    fi
fi

# Test environment summary
echo -e "${YELLOW}=== Test Environment Summary ===${NC}"
echo "Environment Type: production"
echo "Operating System: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
if [ -n "$GPU_COUNT" ]; then
    echo "GPU Count: $GPU_COUNT"
fi
if [ -n "$CUDA_VERSION" ]; then
    echo "CUDA Version: $CUDA_VERSION"
fi
echo "Python Version: $(python --version)"
echo "Packages:"
pip list | grep -E "fastapi|ray|torch|vllm|pytest"

echo -e "${GREEN}=== Test complete ===${NC}"
echo "Coverage report available in: $(pwd)/coverage_report_prod/index.html"