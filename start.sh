#!/bin/bash

# start.sh - Setup script for vLLM Backend & Frontend

# --- Configuration ---
VLLM_USER=$(whoami) # Use current user by default
VLLM_GROUP=$(id -gn "$VLLM_USER") # Get primary group of the user
VLLM_HOME_DEFAULT="/opt/vllm" # Default installation directory
VLLM_HOME="${VLLM_HOME:-$VLLM_HOME_DEFAULT}" # Allow overriding via environment variable
BACKEND_PORT=8080
FRONTEND_PORT=5000
SERVICE_NAME="vllm" # Must match config.py and systemctl_utils.py

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
# Check if running as root (some steps need sudo)
if [[ $EUID -eq 0 ]]; then
  print_warning "Running as root. It's recommended to run as a user with sudo privileges."
  # Consider exiting or prompting
fi

# Check for sudo
if ! command -v sudo &> /dev/null; then
    print_error "sudo command not found. Please install sudo."
    exit 1
fi
# Check if user has sudo privileges (non-interactively)
if ! sudo -n true 2>/dev/null; then
    print_warning "Current user might not have passwordless sudo privileges."
    print_warning "You might be prompted for your password during setup."
    print_warning "Passwordless sudo for 'systemctl [start|stop|restart|enable|disable] ${SERVICE_NAME}' is required for the backend API to fully function."
fi

# Check for NVIDIA drivers (basic check)
if ! command -v nvidia-smi &> /dev/null; then
    print_error "NVIDIA drivers not found (nvidia-smi command missing)."
    print_error "Please install appropriate NVIDIA drivers and CUDA toolkit for your system."
    exit 1
else
    print_info "NVIDIA drivers detected."
    nvidia-smi -L
fi

# Check for Python 3 and venv
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3."
    exit 1
fi
if ! python3 -m venv -h &> /dev/null; then
     print_error "python3-venv module not found. Please install it (e.g., sudo apt install python3-venv)."
     exit 1
fi
print_info "Python 3 and venv found."

# --- System Dependencies ---
print_step "Installing System Dependencies (Requires sudo)"
# Update package list
sudo apt-get update -q || { print_error "apt-get update failed."; exit 1; }
# Install essential tools
sudo apt-get install -y -q build-essential git wget curl jq || { print_error "Failed to install essential packages."; exit 1; }
# Install Python dev headers (needed for some pip packages)
sudo apt-get install -y -q python3-dev || { print_error "Failed to install python3-dev."; exit 1; }
print_success "System dependencies installed."

# --- Create Directories ---
print_step "Creating Installation Directory: ${VLLM_HOME}"
sudo mkdir -p "$VLLM_HOME" || { print_error "Failed to create directory $VLLM_HOME"; exit 1; }
# Ensure the target user owns the directory
sudo chown -R "${VLLM_USER}:${VLLM_GROUP}" "$VLLM_HOME" || { print_error "Failed to change ownership of $VLLM_HOME"; exit 1; }
cd "$VLLM_HOME" || { print_error "Failed to change directory to $VLLM_HOME"; exit 1; }
print_success "Directory created and ownership set."

# --- Setup Python Virtual Environments ---
print_step "Setting up Python Virtual Environments"
# Backend venv
if [ ! -d "venv" ]; then
    python3 -m venv venv || { print_error "Failed to create backend virtual environment."; exit 1; }
    print_info "Backend virtual environment created at ${VLLM_HOME}/venv"
else
    print_info "Backend virtual environment already exists."
fi
# Frontend venv (can share backend venv if desired, but separate is cleaner)
if [ ! -d "frontend_venv" ]; then
    python3 -m venv frontend_venv || { print_error "Failed to create frontend virtual environment."; exit 1; }
    print_info "Frontend virtual environment created at ${VLLM_HOME}/frontend_venv"
else
    print_info "Frontend virtual environment already exists."
fi

# --- Install Python Dependencies ---
print_step "Installing Python Dependencies"
# Backend dependencies
print_info "Installing backend dependencies..."
source "${VLLM_HOME}/venv/bin/activate" || { print_error "Failed to activate backend venv."; exit 1; }
pip install --upgrade pip || print_warning "Failed to upgrade pip."
# Install from requirements.txt if it exists, otherwise install core list
if [ -f "backend/requirements.txt" ]; then
    pip install -r backend/requirements.txt || { print_error "Failed to install backend requirements from file."; exit 1; }
else
    pip install "fastapi[all]" uvicorn huggingface_hub requests pynvml || { print_error "Failed to install core backend dependencies."; exit 1; }
fi
deactivate
print_success "Backend dependencies installed."

# Frontend dependencies
print_info "Installing frontend dependencies..."
source "${VLLM_HOME}/frontend_venv/bin/activate" || { print_error "Failed to activate frontend venv."; exit 1; }
pip install --upgrade pip || print_warning "Failed to upgrade pip."
if [ -f "frontend/requirements.txt" ]; then
    pip install -r frontend/requirements.txt || { print_error "Failed to install frontend requirements from file."; exit 1; }
else
     pip install Flask requests || { print_error "Failed to install core frontend dependencies."; exit 1; }
fi
deactivate
print_success "Frontend dependencies installed."

# --- Setup Systemd Service ---
print_step "Setting up Systemd Service (Requires sudo)"
SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
TEMPLATE_PATH="${VLLM_HOME}/backend/vllm.service.template"

if [ ! -f "$TEMPLATE_PATH" ]; then
    print_error "Systemd service template not found at ${TEMPLATE_PATH}"
    exit 1
fi

print_info "Creating systemd service file at ${SERVICE_FILE_PATH}"
# Replace placeholders in the template and write to the systemd directory
sudo sed -e "s|%USER%|${VLLM_USER}|g" \
         -e "s|%VLLM_HOME%|${VLLM_HOME}|g" \
         "$TEMPLATE_PATH" | sudo tee "$SERVICE_FILE_PATH" > /dev/null || { print_error "Failed to create systemd service file."; exit 1; }

print_info "Reloading systemd daemon..."
sudo systemctl daemon-reload || { print_error "Failed to reload systemd daemon."; exit 1; }

# Optionally enable the service here, or instruct user
# print_info "Enabling vLLM service to start on boot..."
# sudo systemctl enable ${SERVICE_NAME} || print_warning "Failed to enable service automatically."

print_success "Systemd service configured."
print_warning "IMPORTANT: For the backend API to control the service (start/stop/etc.),"
print_warning "the user '${VLLM_USER}' needs passwordless sudo permission for:"
print_warning "'sudo systemctl [start|stop|restart|enable|disable] ${SERVICE_NAME}'"
print_warning "Configure this manually using 'sudo visudo'."
print_warning "Example line for visudo: ${VLLM_USER} ALL=(ALL) NOPASSWD: /bin/systemctl start ${SERVICE_NAME}, /bin/systemctl stop ${SERVICE_NAME}, /bin/systemctl restart ${SERVICE_NAME}, /bin/systemctl enable ${SERVICE_NAME}, /bin/systemctl disable ${SERVICE_NAME}"


# --- Start Services ---
print_step "Starting Backend and Frontend Services"

# Start Backend (uvicorn) in the background
print_info "Starting FastAPI backend on port ${BACKEND_PORT}..."
source "${VLLM_HOME}/venv/bin/activate" || { print_error "Failed to activate backend venv."; exit 1; }
# Run in background, redirect output to log file
nohup uvicorn backend.main:app --host 0.0.0.0 --port ${BACKEND_PORT} --forwarded-allow-ips '*' > "${VLLM_HOME}/logs/backend.log" 2>&1 &
BACKEND_PID=$!
deactivate
sleep 2 # Give it a moment to start
if ps -p $BACKEND_PID > /dev/null; then
   print_success "Backend started successfully (PID: $BACKEND_PID). Logs: ${VLLM_HOME}/logs/backend.log"
else
   print_error "Backend failed to start. Check logs: ${VLLM_HOME}/logs/backend.log"
   # Consider exiting if backend fails
fi

# Start Frontend (Flask) in the background
print_info "Starting Flask frontend on port ${FRONTEND_PORT}..."
source "${VLLM_HOME}/frontend_venv/bin/activate" || { print_error "Failed to activate frontend venv."; exit 1; }
# Run in background, redirect output to log file
nohup python frontend/app.py > "${VLLM_HOME}/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
deactivate
sleep 1
if ps -p $FRONTEND_PID > /dev/null; then
   print_success "Frontend started successfully (PID: $FRONTEND_PID). Logs: ${VLLM_HOME}/logs/frontend.log"
else
   print_error "Frontend failed to start. Check logs: ${VLLM_HOME}/logs/frontend.log"
fi

print_step "Setup Complete!"
echo -e "Backend API should be running at: ${BOLD}http://<server_ip>:${BACKEND_PORT}${NC}"
echo -e "Frontend Admin UI should be running at: ${BOLD}http://<server_ip>:${FRONTEND_PORT}${NC}"
echo -e "Remember to configure passwordless sudo for service control if you haven't already."
echo -e "Use 'sudo systemctl status ${SERVICE_NAME}' to check the vLLM service status."

exit 0