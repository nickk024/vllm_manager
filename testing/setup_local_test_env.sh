#!/bin/bash

# setup_local_test_env.sh - Sets up a local Python venv for testing on macOS/Linux

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

# --- Sanity Checks ---
print_step "Running Sanity Checks"

# Check for pyenv first
RECOMMENDED_PYTHON_VERSION="3.11" # Based on LXC setup logs
PYTHON_CMD="python3" # Default command

if command -v pyenv &> /dev/null; then
    print_info "pyenv detected."
    CURRENT_PY_VERSION=$(pyenv version-name)
    print_info "Current pyenv Python version: ${CURRENT_PY_VERSION}"
    if [[ ! "$CURRENT_PY_VERSION" == *"$RECOMMENDED_PYTHON_VERSION"* ]]; then
        print_warning "Current Python version ($CURRENT_PY_VERSION) is not the recommended ${RECOMMENDED_PYTHON_VERSION}.x."
        print_warning "Consider running 'pyenv install ${RECOMMENDED_PYTHON_VERSION}.x && pyenv local ${RECOMMENDED_PYTHON_VERSION}.x' in this directory."
    fi
    # Use the python managed by pyenv
    PYTHON_CMD="python"
else
    print_info "pyenv not detected. Using default 'python3'."
    if ! command -v python3 &> /dev/null; then print_error "python3 not found. Please install Python 3 or use pyenv."; exit 1; fi
    CURRENT_PY_VERSION=$(python3 --version 2>&1)
    print_info "Default Python version: ${CURRENT_PY_VERSION}"
fi

# Check if the selected python command works and has venv
if ! command -v $PYTHON_CMD &> /dev/null; then print_error "'$PYTHON_CMD' command not found. Please install Python 3 or configure pyenv correctly."; exit 1; fi
if ! $PYTHON_CMD -m venv -h &> /dev/null; then print_error "'$PYTHON_CMD -m venv' module not found. Ensure your Python installation includes venv."; exit 1; fi
print_info "Selected Python ($PYTHON_CMD) and venv module found."

# --- Create Virtual Environment ---
VENV_DIR=".venv_local_test"
print_step "Creating Local Virtual Environment: ${VENV_DIR}"
if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    # Use the determined Python command
    $PYTHON_CMD -m venv "$VENV_DIR" || { print_error "Failed to create virtual environment using '$PYTHON_CMD'."; exit 1; }
    print_success "Virtual environment created."
fi

# --- Install Dependencies ---
print_step "Installing Dependencies into ${VENV_DIR}"
# Activate script for this shell instance to ensure pip is correct one
# source "${VENV_DIR}/bin/activate" # This only works if script is sourced, not run directly. Use direct path to pip instead.

# Define dependencies
# Note: vllm and pynvml might have issues on macOS
dependencies=(
    "pytest"
    "httpx"
    "fastapi[all]"
    "uvicorn"
    "pydantic"
    "ray[serve]"
    "vllm"
    "pynvml"
    "huggingface_hub"
    "requests"
)

print_info "Attempting to install: ${dependencies[*]}"
print_warning "Note: 'vllm' and 'pynvml' installation might fail or have limitations on macOS."

"${VENV_DIR}/bin/pip" install --upgrade pip || print_warning "Failed to upgrade pip."
"${VENV_DIR}/bin/pip" install "${dependencies[@]}"

INSTALL_EXIT_CODE=$?
if [ $INSTALL_EXIT_CODE -ne 0 ]; then
    print_error "Failed to install one or more dependencies (Exit code: $INSTALL_EXIT_CODE)."
    print_warning "Check the output above for specific errors. 'vllm' or 'pynvml' might be the cause on macOS."
    # Optionally exit here, or let the user decide
    # exit 1
else
    print_success "Dependencies installed (or already present)."
fi

print_step "Setup Complete!"
echo -e "To activate the virtual environment, run:"
echo -e "${BOLD}source ${VENV_DIR}/bin/activate${NC}"
echo -e "Once activated, you can run the tests with:"
echo -e "${BOLD}python -m pytest backend/tests/${NC}"
echo -e "Remember to deactivate when finished:"
echo -e "${BOLD}deactivate${NC}"

exit 0