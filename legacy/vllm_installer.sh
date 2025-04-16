#!/bin/bash
# vllm_bare_metal_setup.sh - Enhanced installation script for vLLM
# Creates a complete vLLM installation with more verbosity and user choices

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Script directory (where component scripts are located)
# Get the path of the script
INSTALLER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_DIR="${INSTALLER_DIR}/scripts"

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

print_step() {
    echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}"
}

print_substep() {
    echo -e "${MAGENTA}>>> $1${NC}"
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

# Function to ask for a value with a default
ask_value() {
    local prompt="$1"
    local default="$2"
    local value
    
    read -p "$prompt [$default]: " value
    echo "${value:-$default}"
}

# Function to ask for a choice from options
ask_choice() {
    local prompt="$1"
    shift
    local options=("$@")
    local choice
    
    echo "$prompt"
    for i in "${!options[@]}"; do
        echo "  $((i+1)). ${options[$i]}"
    done
    
    while true; do
        read -p "Enter choice [1-${#options[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            echo "${options[$((choice-1))]}"
            return
        else
            echo "Please enter a number between 1 and ${#options[@]}."
        fi
    done
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a package is installed
package_installed() {
    dpkg -l "$1" | grep -q "^ii" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_step "Checking system requirements"
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_warning "This script is running as root. It's recommended to run as a regular user with sudo privileges."
        if ! ask_yes_no "Continue anyway?" "N"; then
            print_info "Exiting as requested."
            exit 0
        fi
    fi
    
    # Check for sudo access
    if ! command_exists sudo; then
        print_error "sudo is not installed. Please install sudo and try again."
        exit 1
    fi
    
    if ! sudo -n true 2>/dev/null; then
        print_warning "This script requires sudo privileges for some operations."
        print_info "You may be prompted for your password during execution."
    fi
    
    # Check for NVIDIA GPU
    if ! command_exists nvidia-smi; then
        print_error "NVIDIA GPU drivers not detected. vLLM requires an NVIDIA GPU to function."
        print_info "Please install NVIDIA drivers and try again."
        exit 1
    fi
    
    # Display GPU information
    print_substep "Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    # Check for Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_info "Detected Python version: $python_version"
    
    # Check for required commands
    required_commands=("curl" "jq" "git")
    missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -gt 0 ]; then
        print_warning "Some required commands are missing: ${missing_commands[*]}"
        print_info "These will be installed during setup."
    fi
    
    print_success "System requirements check completed."
    return 0
}

# Function to display a spinner for long-running commands
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Function to execute a command with a spinner
execute_with_spinner() {
    local command="$1"
    local message="$2"
    
    echo -n -e "${message} "
    eval "$command" >/dev/null 2>&1 &
    spinner $!
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Function to create model configuration
create_model_config() {
    local config_file="$1"
    local include_llama3="$2"
    local include_mistral="$3"
    local include_phi3="$4"
    local tensor_parallel_size="$5"
    local include_custom_model="$6"
    local custom_model_id="$7"
    local custom_model_name="$8"
    
    print_substep "Creating model configuration at $config_file"
    
    # Start with an empty JSON object
    echo "{" > "$config_file"
    
    # Add models based on user choices
    local first_model=true
    
    if [[ "$include_llama3" == "Y" ]]; then
        cat >> "$config_file" << EOF
    "llama3-8b": {
        "model_id": "meta-llama/Llama-3-8B-Instruct",
        "tensor_parallel_size": $tensor_parallel_size,
        "max_model_len": 8192,
        "dtype": "bfloat16"
    }
EOF
        first_model=false
    fi
    
    if [[ "$include_mistral" == "Y" ]]; then
        if [[ "$first_model" == "false" ]]; then
            echo "," >> "$config_file"
        fi
        cat >> "$config_file" << EOF
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "tensor_parallel_size": $tensor_parallel_size,
        "max_model_len": 8192,
        "dtype": "bfloat16"
    }
EOF
        first_model=false
    fi
    
    if [[ "$include_phi3" == "Y" ]]; then
        if [[ "$first_model" == "false" ]]; then
            echo "," >> "$config_file"
        fi
        cat >> "$config_file" << EOF
    "phi3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "tensor_parallel_size": $tensor_parallel_size,
        "max_model_len": 4096,
        "dtype": "bfloat16"
    }
EOF
        first_model=false
    fi
    
    if [[ "$include_custom_model" == "Y" && -n "$custom_model_id" && -n "$custom_model_name" ]]; then
        if [[ "$first_model" == "false" ]]; then
            echo "," >> "$config_file"
        fi
        cat >> "$config_file" << EOF
    "$custom_model_name": {
        "model_id": "$custom_model_id",
        "tensor_parallel_size": $tensor_parallel_size,
        "max_model_len": 8192,
        "dtype": "bfloat16"
    }
EOF
    fi
    
    # Close the JSON object
    echo "}" >> "$config_file"
    
    print_success "Model configuration created successfully."
}

# Function to install system dependencies
install_dependencies() {
    print_step "Installing dependencies"
    print_substep "Updating package lists"
    sudo apt-get update -q
    
    print_substep "Installing required packages"
    packages="python3-full python3-venv python3-dev build-essential git wget curl jq"
    for pkg in $packages; do
        if ! package_installed "$pkg"; then
            echo -n "Installing $pkg... "
            if sudo apt-get install -y "$pkg" >/dev/null 2>&1; then
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${RED}✗${NC}"
                print_warning "Failed to install $pkg. Continuing anyway."
            fi
        else
            echo -e "Package $pkg is already installed ${GREEN}✓${NC}"
        fi
    done
    
    return 0
}

# Function to set up Python environment
setup_python_env() {
    local venv_dir="$1"
    
    print_step "Setting up Python virtual environment"
    print_substep "Creating virtual environment at $venv_dir"
    
    if python3 -m venv "${venv_dir}"; then
        print_success "Virtual environment created successfully."
    else
        print_error "Failed to create virtual environment."
        exit 1
    fi
    
    print_substep "Activating virtual environment"
    source "${venv_dir}/bin/activate"
    
    print_substep "Upgrading pip"
    execute_with_spinner "pip install --upgrade pip" "Upgrading pip..."
    
    print_substep "Installing PyTorch"
    execute_with_spinner "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" "Installing PyTorch (this may take a while)..."
    
    print_substep "Installing vLLM"
    execute_with_spinner "pip install vllm" "Installing vLLM..."
    
    print_substep "Installing additional dependencies"
    execute_with_spinner "pip install huggingface_hub nvidia-ml-py3 requests" "Installing additional dependencies..."
    
    return 0
}

# Function to copy component scripts
copy_component_scripts() {
    local destination_dir="$1"
    
    print_step "Copying component scripts"
    
    # List of component scripts
    component_scripts=(
        "${SCRIPT_DIR}/vllm/vllm_server.py"
        "${SCRIPT_DIR}/vllm/download_models.py"
        "${SCRIPT_DIR}/vllm/monitor_gpu.py" 
        "${SCRIPT_DIR}/vllm/vllm_manage.sh"
    )
    
    for script in "${component_scripts[@]}"; do
        script_name=$(basename "$script")
        print_substep "Copying $script_name"
        
        if [ -f "$script" ]; then
            cp "$script" "${destination_dir}/${script_name}"
            chmod +x "${destination_dir}/${script_name}"
            print_info "$script_name copied successfully."
        else
            print_warning "$script_name not found at $script."
            
            # Create empty file to avoid errors
            touch "${destination_dir}/${script_name}"
            print_error "Created empty $script_name. You may need to manually copy it later."
        fi
    done
    
    return 0
}

# Function to set up systemd service
setup_systemd_service() {
    local vllm_home="$1"
    local port="$2"
    local user="$3"
    local tp_size="$4"
    local gpu_mem="$5"
    
    print_step "Setting up systemd service"
    
    # Create service file
    local service_file="/etc/systemd/system/vllm.service"
    print_substep "Creating service file at $service_file"
    
    if [ -f "${SCRIPT_DIR}/vllm/vllm_systemd.service" ]; then
        # Use template file
        cat "${SCRIPT_DIR}/vllm/vllm_systemd.service" | \
            sed "s|%USER%|${user}|g" | \
            sed "s|%VLLM_HOME%|${vllm_home}|g" | \
            sed "s|%PORT%|${port}|g" | \
            sed "s|%TP_SIZE%|${tp_size}|g" | \
            sed "s|%GPU_MEM%|${gpu_mem}|g" | \
            sudo tee "$service_file" > /dev/null
    else
        # Create service file directly
        cat <<EOF | sudo tee "$service_file" > /dev/null
[Unit]
Description=vLLM API Server
After=network.target

[Service]
Type=simple
User=${user}
WorkingDirectory=${vllm_home}
Environment="PATH=${vllm_home}/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=${vllm_home}/scripts"
Environment="HF_HOME=${vllm_home}/.cache/huggingface"
ExecStart=${vllm_home}/venv/bin/python ${vllm_home}/scripts/vllm_server.py --host 0.0.0.0 --port ${port} --model-dir ${vllm_home}/models --config ${vllm_home}/config/model_config.json --tensor-parallel-size ${tp_size} --gpu-memory-utilization ${gpu_mem}
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
    fi
    
    print_substep "Reloading systemd daemon"
    sudo systemctl daemon-reload
    
    print_success "Systemd service configured successfully."
    return 0
}

# Main script starts here
clear
echo -e "${BOLD}${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║           Enhanced vLLM Bare Metal Installation            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "This script will install vLLM directly on your system, allowing you"
echo "to run large language models with an OpenAI-compatible API."
echo -e "The enhanced version provides ${BOLD}more verbosity${NC} and ${BOLD}user choices${NC}."
echo

# Check system requirements
check_requirements

# Get user choices
print_step "Configuration Options"

# Installation directory
VLLM_HOME=$(ask_value "Where would you like to install vLLM?" "/opt/vllm")
print_info "vLLM will be installed to: $VLLM_HOME"

# Port number
PORT=$(ask_value "Which port should the vLLM API server use?" "8000")
print_info "vLLM API will be available at: http://localhost:$PORT/v1"

# Model choices
echo
print_substep "Model Selection"

# Offer different model selection approaches
echo -e "${BOLD}How would you like to select models?${NC}"
echo "1. Choose from popular pre-defined models"
echo "2. Browse and select from Hugging Face trending models"
echo "3. Manually specify models"

selection_mode=$(ask_value "Enter selection mode" "1")

# Variables to track selected models
include_llama3="N"
include_mistral="N"
include_phi3="N"
include_custom_model="N"
custom_model_id=""
custom_model_name=""
config_file="${CONFIG_DIR}/model_config.json"
use_popular_models_script="N"

case "$selection_mode" in
    1)
        # Basic pre-defined model selection
        echo "You will need a Hugging Face token for gated models like Llama 3."
        
        if ask_yes_no "Include Llama 3 (8B Instruct)?" "Y"; then
            include_llama3="Y"
        fi
        
        if ask_yes_no "Include Mistral (7B Instruct v0.2)?" "Y"; then
            include_mistral="Y"
        fi
        
        if ask_yes_no "Include Phi-3 (Mini 4K Instruct)?" "Y"; then
            include_phi3="Y"
        fi
        
        if ask_yes_no "Include a custom model from Hugging Face?" "N"; then
            include_custom_model="Y"
            custom_model_id=$(ask_value "Enter the Hugging Face model ID (e.g., EleutherAI/pythia-1.4b)" "")
            if [[ -n "$custom_model_id" ]]; then
                # Extract default model name from ID
                default_name=$(basename "$custom_model_id" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
                custom_model_name=$(ask_value "Enter a short name for this model" "$default_name")
            else
                include_custom_model="N"
            fi
        fi
        ;;
        
    2|3)
        # Advanced model selection using popular_models.py
        use_popular_models_script="Y"
        
        # Check if the popular models script exists
        if [ ! -f "${SCRIPT_DIR}/vllm/popular_models.py" ]; then
            print_warning "Popular models script not found at ${SCRIPT_DIR}/vllm/popular_models.py"
            print_info "Falling back to basic model selection."
            
            if ask_yes_no "Include Llama 3 (8B Instruct)?" "Y"; then
                include_llama3="Y"
            fi
            
            if ask_yes_no "Include Mistral (7B Instruct v0.2)?" "Y"; then
                include_mistral="Y"
            fi
            
            if ask_yes_no "Include Phi-3 (Mini 4K Instruct)?" "Y"; then
                include_phi3="Y"
            fi
        else
            print_info "Using popular models script for advanced model selection."
            print_warning "A complete list of models will be shown later in the setup process."
            
            # May need a HF token for fetching trending models
            if [ "$selection_mode" = "2" ]; then
                HF_TOKEN=$(ask_value "Enter your Hugging Face token (leave empty to continue without token)" "")
            fi
        fi
        ;;
        
    *)
        print_warning "Invalid selection mode. Using basic model selection."
        
        if ask_yes_no "Include Llama 3 (8B Instruct)?" "Y"; then
            include_llama3="Y"
        fi
        
        if ask_yes_no "Include Mistral (7B Instruct v0.2)?" "Y"; then
            include_mistral="Y"
        fi
        
        if ask_yes_no "Include Phi-3 (Mini 4K Instruct)?" "Y"; then
            include_phi3="Y"
        fi
        ;;
esac

# GPU configuration
echo
print_substep "GPU Configuration"

# Get GPU count
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
print_info "Detected $gpu_count GPU(s)"

# Suggest tensor parallel size based on GPU count
default_tp_size=1
if [[ $gpu_count -ge 2 ]]; then
    default_tp_size=2
fi

tensor_parallel_size=$(ask_value "Tensor parallel size (number of GPUs to use per model)" "$default_tp_size")
gpu_memory_util=$(ask_value "GPU memory utilization (0.0-1.0)" "0.85")

# Advanced options
echo
print_substep "Advanced Options"

download_now="N"
if ask_yes_no "Download models after installation?" "Y"; then
    download_now="Y"
fi

# Ask for HuggingFace token if needed and downloading models
if [[ "$download_now" == "Y" && "$include_llama3" == "Y" ]]; then
    echo
    print_warning "Llama 3 models require a Hugging Face token for download."
    HF_TOKEN=$(ask_value "Enter your Hugging Face token (leave empty to skip for now)" "")
    if [[ -z "$HF_TOKEN" ]]; then
        print_info "No token provided. You can set the HF_TOKEN environment variable later."
    fi
fi

# Service configuration
echo
print_substep "Service Configuration"

enable_at_boot="N"
if ask_yes_no "Enable vLLM services to start at boot?" "Y"; then
    enable_at_boot="Y"
fi

start_after_install="N"
if ask_yes_no "Start vLLM services after installation?" "Y"; then
    start_after_install="Y"
fi

# Confirm choices
echo
print_step "Configuration Summary"
echo "Installation directory: $VLLM_HOME"
echo "API port: $PORT"
echo "Models:"
[[ "$include_llama3" == "Y" ]] && echo "  - Llama 3 (8B Instruct)"
[[ "$include_mistral" == "Y" ]] && echo "  - Mistral (7B Instruct v0.2)"
[[ "$include_phi3" == "Y" ]] && echo "  - Phi-3 (Mini 4K Instruct)"
[[ "$include_custom_model" == "Y" ]] && echo "  - $custom_model_name ($custom_model_id)"
echo "Tensor parallel size: $tensor_parallel_size"
echo "GPU memory utilization: $gpu_memory_util"
echo "Download models now: $download_now"
if [[ "$download_now" == "Y" && "$include_llama3" == "Y" ]]; then
    if [[ -z "$HF_TOKEN" ]]; then
        echo "HuggingFace token: Not provided (will need to be set later)"
    else
        echo "HuggingFace token: Provided"
    fi
fi
echo "Enable at boot: $enable_at_boot"
echo "Start after install: $start_after_install"
echo

if ! ask_yes_no "Proceed with installation?" "Y"; then
    print_info "Installation cancelled."
    exit 0
fi

# Set up variables
USER="$(whoami)"
VENV_DIR="${VLLM_HOME}/venv"
MODELS_DIR="${VLLM_HOME}/models"
SCRIPTS_DIR="${VLLM_HOME}/scripts"
LOGS_DIR="${VLLM_HOME}/logs"
CONFIG_DIR="${VLLM_HOME}/config"

# Create required directories
print_step "Creating directories"
print_substep "Creating installation directories"

if sudo mkdir -p "${VLLM_HOME}" "${MODELS_DIR}" "${SCRIPTS_DIR}" "${LOGS_DIR}" "${CONFIG_DIR}"; then
    sudo chown -R ${USER}:${USER} "${VLLM_HOME}"
    print_success "Directories created successfully."
else
    print_error "Failed to create directories. Please check permissions."
    exit 1
fi

# Install system dependencies
install_dependencies

# Set up Python environment
setup_python_env "${VENV_DIR}"

# Create model configuration
print_step "Creating vLLM configuration"
if [ "$use_popular_models_script" = "Y" ]; then
    print_substep "Running popular models selection script"
    
    # Create temporary directory structure for the config
    mkdir -p "${CONFIG_DIR}"
    
    # Copy the popular_models.py script if it doesn't exist in the destination
    if [ ! -f "${SCRIPTS_DIR}/popular_models.py" ]; then
        cp "${SCRIPT_DIR}/vllm/popular_models.py" "${SCRIPTS_DIR}/popular_models.py"
        chmod +x "${SCRIPTS_DIR}/popular_models.py"
    fi
    
    # Build the command
    cmd="${SCRIPTS_DIR}/popular_models.py --output ${config_file} --models-dir ${MODELS_DIR}"
    if [ -n "$HF_TOKEN" ]; then
        cmd="${cmd} --token ${HF_TOKEN}"
    fi
    
    # Run the model selection script
    cd "${SCRIPTS_DIR}"
    python3 "${cmd}"
    
    # Check if the config was created
    if [ ! -f "${config_file}" ]; then
        print_warning "Model configuration creation failed. Using basic configuration."
        create_model_config "${config_file}" "$include_llama3" "$include_mistral" "$include_phi3" "$tensor_parallel_size" "$include_custom_model" "$custom_model_id" "$custom_model_name"
    else
        print_success "Model configuration created using the popular models script."
    fi
else
    # Use the basic model configuration
    create_model_config "${config_file}" "$include_llama3" "$include_mistral" "$include_phi3" "$tensor_parallel_size" "$include_custom_model" "$custom_model_id" "$custom_model_name"
fi

# Copy component scripts
copy_component_scripts "${SCRIPTS_DIR}"

# Set up systemd service
setup_systemd_service "${VLLM_HOME}" "${PORT}" "${USER}" "${tensor_parallel_size}" "${gpu_memory_util}"

# Enable service if requested
if [[ "$enable_at_boot" == "Y" ]]; then
    print_substep "Enabling vLLM service at boot"
    sudo systemctl enable vllm
    print_success "Service enabled successfully."
fi

# Download models if requested
if [[ "$download_now" == "Y" ]]; then
    print_step "Downloading models"
    
    # Prepare command
    download_cmd="${VENV_DIR}/bin/python ${SCRIPTS_DIR}/download_models.py --config ${CONFIG_DIR}/model_config.json --output-dir ${MODELS_DIR}"
    
    # Add token if provided
    if [[ -n "$HF_TOKEN" ]]; then
        download_cmd="${download_cmd} --token ${HF_TOKEN}"
    fi
    
    print_substep "Starting model download"
    echo "This may take a while depending on your internet connection and the models selected."
    
    # Execute the download command
    source "${VENV_DIR}/bin/activate"
    eval "$download_cmd"
    
    if [[ $? -eq 0 ]]; then
        print_success "Models downloaded successfully."
    else
        print_warning "There were issues downloading some models."
        print_info "You can try downloading them later using the management script."
    fi
fi

# Start service if requested
if [[ "$start_after_install" == "Y" ]]; then
    print_substep "Starting vLLM service"
    sudo systemctl start vllm
    sleep 2
    
    if systemctl is-active --quiet vllm; then
        print_success "vLLM service started successfully."
    else
        print_warning "Failed to start vLLM service. You can try starting it manually later."
    fi
fi

# Create a symlink to management script
if command_exists sudo; then
    print_substep "Creating symlink to management script"
    sudo ln -sf "${SCRIPTS_DIR}/vllm_manage.sh" "/usr/local/bin/vllm-manage"
    sudo chmod +x "/usr/local/bin/vllm-manage"
    print_success "Management script symlinked to /usr/local/bin/vllm-manage"
fi

# Installation complete
print_step "Installation Complete"
echo "vLLM has been successfully installed on your system."
echo
echo "API Endpoint: http://localhost:${PORT}/v1"
echo "Installation Directory: ${VLLM_HOME}"
echo "Models Directory: ${MODELS_DIR}"
echo "Logs Directory: ${LOGS_DIR}"
echo
echo -e "${BOLD}Management Commands:${NC}"
echo "  vllm-manage help                  # Show all available commands"
echo "  vllm-manage status                # Check service status"
echo "  vllm-manage list-models           # List available models"
echo "  vllm-manage download-models       # Download all configured models"
echo "  vllm-manage monitor               # Monitor GPU usage"
echo "  vllm-manage test                  # Test the API"
echo

# If Llama3 was selected but no token was provided
if [[ "$include_llama3" == "Y" && -z "$HF_TOKEN" ]]; then
    print_warning "You selected Llama 3 but didn't provide a Hugging Face token."
    print_info "To download Llama 3 models, you'll need to set the HF_TOKEN environment variable:"
    echo "  export HF_TOKEN=your_huggingface_token"
    echo "  vllm-manage download-models"
    echo
fi

print_success "Thank you for installing vLLM!"
exit 0
