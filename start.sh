#!/bin/bash

# start.sh - Setup script for vLLM Backend & Frontend

# --- Configuration ---
# Script assumes it's run as root in the target LXC environment
VLLM_USER="root" # Running as root
VLLM_GROUP="root" # Running as root
VLLM_HOME_DEFAULT="/opt/vllm" # Default installation directory
VLLM_HOME="${VLLM_HOME:-$VLLM_HOME_DEFAULT}" # Allow overriding via environment variable
BACKEND_PORT=8080
FRONTEND_PORT=5000
SERVICE_NAME="vllm" # Must match config.py and systemctl_utils.py

# Get the directory where this script is located (project root)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/logs" # Log directory in project root

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
if [[ $EUID -ne 0 ]]; then
  print_error "This script expects to be run as root in the target LXC."
  # exit 1 # Or prompt? For now, just warn.
  print_warning "Running as non-root. Some commands might fail."
  VLLM_USER=$(whoami) # Adjust user if not root
  VLLM_GROUP=$(id -gn "$VLLM_USER")
fi

# Check for required commands (sudo not needed if running as root)
if ! command -v nvidia-smi &> /dev/null; then print_error "NVIDIA drivers not found (nvidia-smi missing)."; exit 1; fi
print_info "NVIDIA drivers detected."; nvidia-smi -L
if ! command -v python3 &> /dev/null; then print_error "python3 not found."; exit 1; fi
if ! python3 -m venv -h &> /dev/null; then print_error "python3-venv module not found."; exit 1; fi
print_info "Python 3 and venv found."
if ! command -v rsync &> /dev/null; then print_warning "rsync command not found. Will attempt to install."; fi
print_info "Checks complete."


# --- Optional Cleanup ---
print_step "Checking for Existing Installation"
PERFORM_CLEANUP=false
if [ -d "$VLLM_HOME" ] || [ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]; then
    print_warning "An existing installation directory (${VLLM_HOME}) or service file seems to exist."
    if ask_yes_no "Do you want to remove the existing installation and logs before proceeding?" "N"; then
        PERFORM_CLEANUP=true
    else
        print_info "Proceeding without cleanup. Files might be overwritten."
    fi
else
    print_info "No existing installation found at ${VLLM_HOME} or systemd path."
fi

# Function to ask yes/no questions (defined here for cleanup prompt)
ask_yes_no() {
    local prompt="$1"
    local default="${2:-N}" # Default to No if not specified

    if [[ "$default" =~ ^[Yy]$ ]]; then
        prompt="$prompt [Y/n]"
        default="Y"
    else
        prompt="$prompt [y/N]"
        default="N"
    fi

    while true; do
        read -p "$prompt " choice
        choice=${choice:-$default} # Use default if user presses Enter
        case "$choice" in
            [Yy]* ) return 0;; # Yes returns 0 (success in shell)
            [Nn]* ) return 1;; # No returns 1 (failure in shell)
            * ) echo "Please answer yes or no.";;
        esac
    done
}


if [ "$PERFORM_CLEANUP" = true ]; then
    print_step "Performing Cleanup"
    print_info "Stopping service ${SERVICE_NAME}..."
    systemctl stop "$SERVICE_NAME" # No sudo needed if root
    print_info "Disabling service ${SERVICE_NAME}..."
    systemctl disable "$SERVICE_NAME" # No sudo needed if root
    SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
    if [ -f "$SERVICE_FILE_PATH" ]; then
        print_info "Removing service file ${SERVICE_FILE_PATH}..."
        rm -f "$SERVICE_FILE_PATH" || print_warning "Failed to remove service file." # No sudo
    fi
    print_info "Reloading systemd daemon..."
    systemctl daemon-reload || print_warning "Failed to reload systemd daemon." # No sudo
    print_info "Removing installation directory ${VLLM_HOME}..."
    rm -rf "$VLLM_HOME" || print_warning "Failed to remove installation directory." # No sudo
    print_info "Removing log directory ${LOG_DIR}..."
    rm -rf "$LOG_DIR" || print_warning "Failed to remove log directory."
    print_warning "Cleanup might not stop manually started backend/frontend processes."
    print_success "Cleanup finished."
fi


# --- System Dependencies ---
print_step "Installing System Dependencies"
apt-get update -q || { print_error "apt-get update failed."; exit 1; }
apt-get install -y -q build-essential git wget curl jq rsync python3-dev python3-venv || { print_error "Failed to install essential packages."; exit 1; }
print_success "System dependencies installed."

# --- Create Directories & Copy Project Files ---
print_step "Creating Installation Directory: ${VLLM_HOME}"
mkdir -p "$VLLM_HOME" || { print_error "Failed to create directory $VLLM_HOME"; exit 1; }
chown -R "${VLLM_USER}:${VLLM_GROUP}" "$VLLM_HOME" || { print_error "Failed to change ownership of $VLLM_HOME"; exit 1; } # Keep chown for consistency if run as non-root later
print_success "Directory created and ownership set."

print_step "Copying Project Files to ${VLLM_HOME}"
rsync -a --delete --exclude='.git/' --exclude='legacy/' --exclude='logs/' --exclude='*venv/' --exclude='__pycache__/' "${SCRIPT_DIR}/" "${VLLM_HOME}/" || { print_error "Failed to copy project files using rsync."; exit 1; }
chown -R "${VLLM_USER}:${VLLM_GROUP}" "$VLLM_HOME" || { print_error "Failed to set final ownership of $VLLM_HOME"; exit 1; }
print_success "Project files copied."

# --- Create Log Directory ---
print_step "Creating Log Directory: ${LOG_DIR}"
mkdir -p "$LOG_DIR" || { print_error "Failed to create log directory $LOG_DIR"; exit 1; }
chown "${VLLM_USER}:${VLLM_GROUP}" "$LOG_DIR" || print_warning "Could not set ownership of log directory ${LOG_DIR}"
print_success "Log directory created."

# --- Change into Installation Directory ---
print_info "Attempting to change directory to ${VLLM_HOME}"
cd "$VLLM_HOME"
CD_EXIT_CODE=$?
if [ $CD_EXIT_CODE -ne 0 ]; then
    print_error "Failed to change directory to $VLLM_HOME (Exit code: $CD_EXIT_CODE)"
    exit 1
fi
print_info "Successfully changed working directory to $(pwd)"


# --- Setup Python Virtual Environments ---
print_step "Setting up Python Virtual Environments (in $(pwd))"
if [ ! -d "venv" ]; then
    python3 -m venv venv || { print_error "Failed to create backend virtual environment."; exit 1; }
    print_info "Backend virtual environment created."
else
    print_info "Backend virtual environment already exists."
fi
if [ ! -d "frontend_venv" ]; then
    python3 -m venv frontend_venv || { print_error "Failed to create frontend virtual environment."; exit 1; }
    print_info "Frontend virtual environment created."
else
    print_info "Frontend virtual environment already exists."
fi

# --- Install Python Dependencies ---
print_step "Installing Python Dependencies"
# Backend dependencies
print_info "Installing backend dependencies..."
# Use absolute path to pip in venv
"${VLLM_HOME}/venv/bin/pip" install --upgrade pip || print_warning "Failed to upgrade pip."
if [ ! -f "backend/requirements.txt" ]; then
    print_warning "backend/requirements.txt not found. Installing core list."
    "${VLLM_HOME}/venv/bin/pip" install "fastapi[all]" uvicorn huggingface_hub requests pynvml || { print_error "Failed to install core backend dependencies."; exit 1; }
else
    "${VLLM_HOME}/venv/bin/pip" install -r backend/requirements.txt || { print_error "Failed to install backend requirements from file."; exit 1; }
fi
print_success "Backend dependencies installed."

# Frontend dependencies
print_info "Installing frontend dependencies..."
"${VLLM_HOME}/frontend_venv/bin/pip" install --upgrade pip || print_warning "Failed to upgrade pip."
if [ ! -f "frontend/requirements.txt" ]; then
     print_warning "frontend/requirements.txt not found. Installing core list."
     "${VLLM_HOME}/frontend_venv/bin/pip" install Flask requests || { print_error "Failed to install core frontend dependencies."; exit 1; }
else
    "${VLLM_HOME}/frontend_venv/bin/pip" install -r frontend/requirements.txt || { print_error "Failed to install frontend requirements from file."; exit 1; }
fi
print_success "Frontend dependencies installed."

# --- Setup Systemd Service ---
print_step "Setting up Systemd Service"
SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
TEMPLATE_PATH="${VLLM_HOME}/backend/vllm.service.template"

print_info "Checking for template at: ${TEMPLATE_PATH}"
if [ ! -f "$TEMPLATE_PATH" ]; then
    print_error "Systemd service template NOT FOUND at ${TEMPLATE_PATH}"
    print_info "Listing contents of ${VLLM_HOME}/backend:"
    ls -la "${VLLM_HOME}/backend"
    exit 1
else
    print_success "Systemd service template found."
fi

print_info "Creating systemd service file at ${SERVICE_FILE_PATH}"
# Use cat and pipe to tee instead of sed for simplicity if only replacing known placeholders
# Ensure VLLM_USER is correctly substituted if script wasn't run as root initially
EFFECTIVE_USER=$VLLM_USER # Use the determined user
cat "$TEMPLATE_PATH" | sed -e "s|%USER%|${EFFECTIVE_USER}|g" -e "s|%VLLM_HOME%|${VLLM_HOME}|g" | tee "$SERVICE_FILE_PATH" > /dev/null || { print_error "Failed to create systemd service file."; exit 1; }

print_info "Reloading systemd daemon..."
systemctl daemon-reload || { print_error "Failed to reload systemd daemon."; exit 1; } # No sudo needed

print_success "Systemd service configured."
# Remove sudo warning if running as root
if [[ $EUID -ne 0 ]]; then
    print_warning "IMPORTANT: For the backend API to control the service (start/stop/etc.),"
    print_warning "the user '${EFFECTIVE_USER}' needs passwordless sudo permission for:"
    print_warning "'sudo systemctl [start|stop|restart|enable|disable] ${SERVICE_NAME}'"
    print_warning "Configure this manually using 'sudo visudo'."
    print_warning "Example line for visudo: ${EFFECTIVE_USER} ALL=(ALL) NOPASSWD: /bin/systemctl start ${SERVICE_NAME}, /bin/systemctl stop ${SERVICE_NAME}, /bin/systemctl restart ${SERVICE_NAME}, /bin/systemctl enable ${SERVICE_NAME}, /bin/systemctl disable ${SERVICE_NAME}"
else
    print_info "Running as root, passwordless sudo for systemctl commands is not required."
fi


# --- Start Services ---
print_step "Starting Backend and Frontend Services"

# Start Backend (uvicorn) in the background using absolute paths
print_info "Starting FastAPI backend on port ${BACKEND_PORT}..."
export VLLM_LOG_DIR="${LOG_DIR}" # Pass log dir via env var
nohup "${VLLM_HOME}/venv/bin/python" -m uvicorn backend.main:app --host 0.0.0.0 --port ${BACKEND_PORT} --forwarded-allow-ips '*' > "${LOG_DIR}/backend_stdout.log" 2>&1 &
BACKEND_PID=$!
unset VLLM_LOG_DIR
sleep 2
if ps -p $BACKEND_PID > /dev/null; then
   print_success "Backend started successfully (PID: $BACKEND_PID). Logs: ${LOG_DIR}/vllm_backend.log"
else
   print_error "Backend failed to start. Check logs: ${LOG_DIR}/backend_stdout.log and ${LOG_DIR}/vllm_backend.log"
fi

# Start Frontend (Flask) in the background using absolute paths
print_info "Starting Flask frontend on port ${FRONTEND_PORT}..."
export VLLM_LOG_DIR="${LOG_DIR}" # Pass log dir via env var
export FLASK_APP=frontend/app.py # Relative path should be okay here as we cd'd into VLLM_HOME
nohup "${VLLM_HOME}/frontend_venv/bin/python" -m flask run --host 0.0.0.0 --port ${FRONTEND_PORT} > "${LOG_DIR}/frontend_stdout.log" 2>&1 &
FRONTEND_PID=$!
unset VLLM_LOG_DIR
sleep 1
if ps -p $FRONTEND_PID > /dev/null; then
   print_success "Frontend started successfully (PID: $FRONTEND_PID). Logs: ${LOG_DIR}/vllm_frontend.log"
else
   print_error "Frontend failed to start. Check logs: ${LOG_DIR}/frontend_stdout.log and ${LOG_DIR}/vllm_frontend.log"
fi

print_step "Setup Complete!"
echo -e "Backend API should be running at: ${BOLD}http://<server_ip>:${BACKEND_PORT}${NC}"
echo -e "Frontend Admin UI should be running at: ${BOLD}http://<server_ip>:${FRONTEND_PORT}${NC}"
# Adjust sudo warning based on execution user
if [[ $EUID -ne 0 ]]; then
    echo -e "Remember to configure passwordless sudo for service control if you haven't already."
fi
echo -e "Use 'systemctl status ${SERVICE_NAME}' to check the vLLM service status (it may need to be started manually or via the UI)."

exit 0