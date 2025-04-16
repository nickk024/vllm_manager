#!/bin/bash
# vllm_manage.sh - Management script for vLLM installation

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default values
VLLM_HOME="${VLLM_HOME:-/opt/vllm}"
CONFIG_DIR="${VLLM_HOME}/config"
MODELS_DIR="${VLLM_HOME}/models"
LOGS_DIR="${VLLM_HOME}/logs"
SCRIPTS_DIR="${VLLM_HOME}/scripts"
SERVICE_NAME="vllm"

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

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This command must be run as root."
        exit 1
    fi
}

# Function to check if vLLM is installed
check_vllm_installed() {
    if [[ ! -d "${VLLM_HOME}" ]]; then
        print_error "vLLM installation not found at ${VLLM_HOME}."
        print_info "Please set the VLLM_HOME environment variable or install vLLM first."
        exit 1
    fi
}

# Function to ensure the Python virtual environment is active
ensure_venv() {
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        print_info "Activating virtual environment..."
        source "${VLLM_HOME}/venv/bin/activate"
    fi
}

# Function to start the vLLM service
start_service() {
    check_root
    print_section "Starting vLLM Service"
    
    systemctl start ${SERVICE_NAME}
    if [[ $? -eq 0 ]]; then
        print_success "vLLM service started successfully."
        print_info "Check status with: vllm_manage.sh status"
    else
        print_error "Failed to start vLLM service. Check systemd logs."
        print_info "systemctl status ${SERVICE_NAME}"
    fi
}

# Function to stop the vLLM service
stop_service() {
    check_root
    print_section "Stopping vLLM Service"
    
    systemctl stop ${SERVICE_NAME}
    if [[ $? -eq 0 ]]; then
        print_success "vLLM service stopped successfully."
    else
        print_error "Failed to stop vLLM service. Check systemd logs."
        print_info "systemctl status ${SERVICE_NAME}"
    fi
}

# Function to restart the vLLM service
restart_service() {
    check_root
    print_section "Restarting vLLM Service"
    
    systemctl restart ${SERVICE_NAME}
    if [[ $? -eq 0 ]]; then
        print_success "vLLM service restarted successfully."
        print_info "Check status with: vllm_manage.sh status"
    else
        print_error "Failed to restart vLLM service. Check systemd logs."
        print_info "systemctl status ${SERVICE_NAME}"
    fi
}

# Function to check the status of the vLLM service
check_status() {
    print_section "vLLM Service Status"
    
    systemctl status ${SERVICE_NAME}
}

# Function to enable the vLLM service at boot
enable_service() {
    check_root
    print_section "Enabling vLLM Service at Boot"
    
    systemctl enable ${SERVICE_NAME}
    if [[ $? -eq 0 ]]; then
        print_success "vLLM service enabled at boot successfully."
    else
        print_error "Failed to enable vLLM service at boot. Check systemd logs."
    fi
}

# Function to disable the vLLM service at boot
disable_service() {
    check_root
    print_section "Disabling vLLM Service at Boot"
    
    systemctl disable ${SERVICE_NAME}
    if [[ $? -eq 0 ]]; then
        print_success "vLLM service disabled at boot successfully."
    else
        print_error "Failed to disable vLLM service at boot. Check systemd logs."
    fi
}

# Function to list available models
list_models() {
    check_vllm_installed
    print_section "Available Models"
    
    if [[ -f "${CONFIG_DIR}/model_config.json" ]]; then
        print_info "Models configured in model_config.json:"
        cat "${CONFIG_DIR}/model_config.json" | jq -r 'keys[]' | while read -r model; do
            if [[ -d "${MODELS_DIR}/${model}" ]]; then
                echo -e "  - ${GREEN}${model}${NC} (downloaded)"
            else
                echo -e "  - ${YELLOW}${model}${NC} (not downloaded)"
            fi
        done
    else
        print_warning "No model configuration found at ${CONFIG_DIR}/model_config.json."
        print_info "Available models in ${MODELS_DIR}:"
        if [[ -d "${MODELS_DIR}" ]] && [[ "$(ls -A "${MODELS_DIR}")" ]]; then
            ls -1 "${MODELS_DIR}"
        else
            print_info "No models found."
        fi
    fi
}

# Function to download models
download_models() {
    check_vllm_installed
    ensure_venv
    print_section "Downloading Models"
    
    if [[ ! -f "${CONFIG_DIR}/model_config.json" ]]; then
        print_error "Model configuration not found at ${CONFIG_DIR}/model_config.json."
        exit 1
    fi
    
    # Check for HF_TOKEN if downloading llama3 models
    if grep -q "meta-llama" "${CONFIG_DIR}/model_config.json"; then
        if [[ -z "${HF_TOKEN}" ]]; then
            print_warning "Models include Meta-Llama models, which require a Hugging Face token."
            print_info "Please set the HF_TOKEN environment variable:"
            print_info "export HF_TOKEN=your_token_here"
            
            read -p "Do you want to continue without a token? (y/N) " choice
            if [[ ! $choice =~ ^[Yy]$ ]]; then
                print_info "Aborting download."
                exit 0
            fi
        fi
    fi
    
    # Download all models from the configuration
    python "${SCRIPTS_DIR}/download_models.py" \
        --config "${CONFIG_DIR}/model_config.json" \
        --output-dir "${MODELS_DIR}" \
        --token "${HF_TOKEN}"
}

# Function to download a specific model
download_model() {
    check_vllm_installed
    ensure_venv
    print_section "Downloading Specific Model"
    
    if [[ ! -f "${CONFIG_DIR}/model_config.json" ]]; then
        print_error "Model configuration not found at ${CONFIG_DIR}/model_config.json."
        exit 1
    fi
    
    model_name="$1"
    
    # Check if model exists in configuration
    if ! cat "${CONFIG_DIR}/model_config.json" | jq -e --arg model "$model_name" '.[$model]' > /dev/null; then
        print_error "Model '$model_name' not found in configuration."
        print_info "Available models:"
        cat "${CONFIG_DIR}/model_config.json" | jq -r 'keys[]'
        exit 1
    fi
    
    # Check for HF_TOKEN if downloading llama3 models
    if grep -q "meta-llama" "${CONFIG_DIR}/model_config.json"; then
        if [[ -z "${HF_TOKEN}" ]]; then
            print_warning "Models include Meta-Llama models, which require a Hugging Face token."
            print_info "Please set the HF_TOKEN environment variable:"
            print_info "export HF_TOKEN=your_token_here"
            
            read -p "Do you want to continue without a token? (y/N) " choice
            if [[ ! $choice =~ ^[Yy]$ ]]; then
                print_info "Aborting download."
                exit 0
            fi
        fi
    fi
    
    # Download specific model
    python "${SCRIPTS_DIR}/download_models.py" \
        --config "${CONFIG_DIR}/model_config.json" \
        --output-dir "${MODELS_DIR}" \
        --models "$model_name" \
        --token "${HF_TOKEN}"
}

# Function to monitor GPU usage
monitor_gpu() {
    check_vllm_installed
    ensure_venv
    print_section "GPU Monitoring"
    
    log_file="${LOGS_DIR}/gpu_stats_$(date +%Y%m%d_%H%M%S).log"
    interval=${1:-30}  # Default to 30 seconds if not specified
    
    print_info "Starting GPU monitoring every ${interval} seconds."
    print_info "Metrics will be saved to ${log_file}"
    print_info "Press Ctrl+C to stop monitoring."
    
    python "${SCRIPTS_DIR}/monitor_gpu.py" \
        --output "${log_file}" \
        --interval "${interval}" \
        --dashboard
}

# Function to test API
test_api() {
    check_vllm_installed
    print_section "Testing vLLM API"
    
    port=${1:-8000}  # Default to port 8000 if not specified
    url="http://localhost:${port}/v1/models"
    
    print_info "Testing vLLM API at ${url}"
    print_info "Response:"
    curl -s "${url}" | jq .
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to connect to vLLM API. Make sure the service is running."
        print_info "Check status with: vllm_manage.sh status"
    fi
}

# Function to display help text
display_help() {
    cat << EOF
${BOLD}vLLM Management Script${NC}

Usage: vllm_manage.sh [command] [options]

Commands:
  start              Start the vLLM service
  stop               Stop the vLLM service
  restart            Restart the vLLM service
  status             Check the status of the vLLM service
  enable             Enable the vLLM service at boot
  disable            Disable the vLLM service at boot
  list-models        List available models
  download-models    Download all models from the configuration
  download-model     Download a specific model
  monitor            Monitor GPU usage
  test               Test the vLLM API
  help               Display this help text

Examples:
  vllm_manage.sh start
  vllm_manage.sh download-model llama3-8b
  vllm_manage.sh monitor 10
  vllm_manage.sh test 8000

Environment Variables:
  VLLM_HOME          Path to vLLM installation (default: /opt/vllm)
  HF_TOKEN           Hugging Face token for downloading gated models
EOF
}

# Main function
main() {
    case "$1" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            check_status
            ;;
        enable)
            enable_service
            ;;
        disable)
            disable_service
            ;;
        list-models)
            list_models
            ;;
        download-models)
            download_models
            ;;
        download-model)
            if [[ -z "$2" ]]; then
                print_error "Please specify a model name."
                print_info "Usage: vllm_manage.sh download-model [model_name]"
                exit 1
            fi
            download_model "$2"
            ;;
        monitor)
            monitor_gpu "$2"
            ;;
        test)
            test_api "$2"
            ;;
        help)
            display_help
            ;;
        *)
            print_error "Unknown command: $1"
            display_help
            exit 1
            ;;
    esac
}

# If no arguments, display help
if [[ $# -eq 0 ]]; then
    display_help
    exit 0
fi

# Execute main function with all arguments
main "$@"
