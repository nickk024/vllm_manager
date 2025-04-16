#!/bin/bash
# install_models.sh - Interactive script to select and install models for vLLM

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Defaults
VLLM_HOME="${VLLM_HOME:-/opt/vllm}"
MODELS_DIR="${VLLM_HOME}/models"
CONFIG_DIR="${VLLM_HOME}/config"
CONFIG_FILE="${CONFIG_DIR}/model_config.json"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}"
}

# Function to ask yes/no questions
ask_yes_no() {
    local prompt="$1"
    local default="$2"
    
    if [[ "$default" == "Y" ]]; then
        prompt="$prompt [Y/n]"
        default="Y"
    else
        prompt="$prompt [y/N]"
        default="N"
    fi
    
    while true; do
        read -p "$prompt " choice
        choice=${choice:-$default}
        case "$choice" in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check requirements
print_section "Checking Requirements"

# Check for Python
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check for required Python packages
print_info "Checking for required Python packages..."

python3 -c "import json, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Missing required Python packages. Installing..."
    pip3 install requests 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Failed to install required packages. Please run: pip install requests"
    fi
fi

# Check popular_models.py script
if [ ! -f "${SCRIPT_DIR}/popular_models.py" ]; then
    print_error "Required script not found: ${SCRIPT_DIR}/popular_models.py"
    print_info "Please make sure you have the complete vLLM installation package."
    exit 1
fi
chmod +x "${SCRIPT_DIR}/popular_models.py"

# Check if models directory exists
if [ ! -d "${MODELS_DIR}" ]; then
    print_warning "Models directory not found: ${MODELS_DIR}"
    print_info "Creating models directory..."
    
    mkdir -p "${MODELS_DIR}"
    if [ $? -ne 0 ]; then
        print_error "Failed to create models directory. Please check permissions."
        print_info "You can set the VLLM_HOME environment variable to change the installation directory."
        exit 1
    fi
fi

# Check if config directory exists
if [ ! -d "${CONFIG_DIR}" ]; then
    print_info "Creating config directory: ${CONFIG_DIR}"
    mkdir -p "${CONFIG_DIR}"
fi

# Display welcome message
clear
echo -e "${BOLD}${GREEN}"
echo "╔═════════════════════════════════════════════════╗"
echo "║                                                 ║"
echo "║         vLLM Model Selection & Installation     ║"
echo "║                                                 ║"
echo "╚═════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "This tool helps you select and install models for vLLM."
echo "You can select models from a curated list of popular models"
echo "or browse trending models on Hugging Face."
echo

# Offer selection modes
print_section "Model Selection Mode"
echo "1. Browse and select from categorized popular models"
echo "2. Browse trending models from Hugging Face"
echo "3. Use existing model configuration (if available)"
echo

selection_mode=""
while [[ ! "$selection_mode" =~ ^[1-3]$ ]]; do
    read -p "Enter your choice [1-3]: " selection_mode
done

# Handle HF token
HF_TOKEN=""
if [[ "$selection_mode" != "3" ]]; then
    print_section "Hugging Face Token"
    echo "Some models (like Llama 3) require a Hugging Face token to download."
    echo "You can get one from: https://huggingface.co/settings/tokens"
    
    if ask_yes_no "Do you have a Hugging Face token to use?" "N"; then
        read -p "Enter your Hugging Face token: " HF_TOKEN
        print_success "Token provided. Models requiring authentication will be available."
    else
        print_warning "No token provided. Gated models will be visible but not downloadable."
    fi
fi

# Set up Python environment if using venv
if [ -d "${VLLM_HOME}/venv" ] && [ -f "${VLLM_HOME}/venv/bin/activate" ]; then
    print_info "Using vLLM Python environment."
    source "${VLLM_HOME}/venv/bin/activate"
fi

# Run appropriate command based on selection mode
case "$selection_mode" in
    1)
        print_section "Popular Models"
        
        # Run the popular_models.py script
        python3 "${SCRIPT_DIR}/popular_models.py" --output "${CONFIG_FILE}" --models-dir "${MODELS_DIR}" ${HF_TOKEN:+--token "$HF_TOKEN"}
        
        if [ ! -f "${CONFIG_FILE}" ]; then
            print_error "Failed to create model configuration. Exiting."
            exit 1
        fi
        ;;
    2)
        print_section "Trending Models from Hugging Face"
        
        # Run with trending models option
        python3 "${SCRIPT_DIR}/popular_models.py" --output "${CONFIG_FILE}" --models-dir "${MODELS_DIR}" ${HF_TOKEN:+--token "$HF_TOKEN"} --trending
        
        if [ ! -f "${CONFIG_FILE}" ]; then
            print_error "Failed to create model configuration. Exiting."
            exit 1
        fi
        ;;
    3)
        print_section "Using Existing Configuration"
        
        if [ ! -f "${CONFIG_FILE}" ]; then
            print_error "No existing configuration found at: ${CONFIG_FILE}"
            print_info "Please choose another option or run vLLM installation first."
            exit 1
        fi
        
        print_success "Found existing model configuration."
        echo "Models in configuration:"
        
        # Display models in the configuration
        python3 -c "
import json
try:
    with open('${CONFIG_FILE}') as f:
        config = json.load(f)
    print('\n'.join(f'  - {name} ({info.get(\"model_id\", \"Unknown\")})' for name, info in config.items()))
except Exception as e:
    print(f'Error reading configuration: {e}')
"
        ;;
esac

# Ask if user wants to download the models
print_section "Download Models"

if ask_yes_no "Would you like to download the selected models now?" "Y"; then
    print_info "Starting model download..."
    
    # Check if download_models.py exists
    download_script="${VLLM_HOME}/scripts/download_models.py"
    if [ ! -f "$download_script" ]; then
        # Try to copy from current directory
        if [ -f "${SCRIPT_DIR}/download_models.py" ]; then
            cp "${SCRIPT_DIR}/download_models.py" "${VLLM_HOME}/scripts/"
            chmod +x "${VLLM_HOME}/scripts/download_models.py"
        else
            print_error "Download script not found. Please run the vLLM installation first."
            exit 1
        fi
    fi
    
    # Build download command
    download_cmd="python3 ${download_script} --config ${CONFIG_FILE} --output-dir ${MODELS_DIR}"
    if [ -n "$HF_TOKEN" ]; then
        download_cmd="${download_cmd} --token ${HF_TOKEN}"
    fi
    
    # Execute download
    eval "$download_cmd"
    
    if [ $? -eq 0 ]; then
        print_success "Models downloaded successfully."
    else
        print_warning "There were issues downloading some models."
        print_info "Check the logs above for details."
    fi
else
    print_info "Models not downloaded. You can download them later with:"
    echo "  python ${VLLM_HOME}/scripts/download_models.py --config ${CONFIG_FILE} --output-dir ${MODELS_DIR}${HF_TOKEN:+ --token YOUR_TOKEN}"
fi

# Finished
print_section "Installation Complete"
print_success "Model configuration has been set up successfully."
echo
echo -e "${BOLD}Summary:${NC}"
echo "Configuration file: ${CONFIG_FILE}"
echo "Models directory: ${MODELS_DIR}"
echo
echo -e "${BOLD}To use vLLM with these models:${NC}"
echo "1. Ensure the vLLM service is running"
echo "2. Access the API at http://localhost:8000/v1 (default port)"
echo "3. Use the model names as defined in the configuration"
echo
print_info "Thank you for using vLLM!"

exit 0
