#!/bin/bash

# start.sh - Setup script for vLLM Backend & Frontend

# --- Configuration ---
VLLM_USER="root" # Running as root
VLLM_GROUP="root" # Running as root
VLLM_HOME_DEFAULT="/opt/vllm" # Default installation directory
VLLM_HOME="${VLLM_HOME:-$VLLM_HOME_DEFAULT}" # Allow overriding via environment variable
BACKEND_PORT=8080
FRONTEND_PORT=5000
SERVICE_NAME="vllm" # Must match config.py and systemctl_utils.py
BACKEND_PID_FILE="${VLLM_HOME}/backend.pid" # PID file location
FRONTEND_PID_FILE="${VLLM_HOME}/frontend.pid" # PID file location

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

# Function to ask yes/no questions
ask_yes_no() {
    local prompt="$1"
    local default="${2:-N}" # Default to No if not specified
    if [[ "$default" =~ ^[Yy]$ ]]; then prompt="$prompt [Y/n]"; default="Y"; else prompt="$prompt [y/N]"; default="N"; fi
    while true; do read -p "$prompt " choice; choice=${choice:-$default}; case "$choice" in [Yy]* ) return 0;; [Nn]* ) return 1;; * ) echo "Please answer yes or no.";; esac; done
}

# --- Sanity Checks ---
print_step "Running Sanity Checks"
if [[ $EUID -ne 0 ]]; then print_error "This script expects to be run as root."; exit 1; fi
if ! command -v nvidia-smi &> /dev/null; then print_error "NVIDIA drivers not found."; exit 1; fi
print_info "NVIDIA drivers detected."; nvidia-smi -L
if ! command -v python3 &> /dev/null; then print_error "python3 not found."; exit 1; fi
if ! python3 -m venv -h &> /dev/null; then print_error "python3-venv module not found."; exit 1; fi
print_info "Python 3 and venv found."
if ! command -v rsync &> /dev/null; then print_warning "rsync command not found. Will attempt to install."; fi
if ! command -v jq &> /dev/null; then print_warning "jq command not found. Will attempt to install."; fi # Add jq check
print_info "Checks complete."


# --- Optional Cleanup ---
print_step "Checking for Existing Installation"
PERFORM_CLEANUP=false
if [ -d "$VLLM_HOME" ] || [ -f "/etc/systemd/system/${SERVICE_NAME}.service" ] || [ -f "$BACKEND_PID_FILE" ] || [ -f "$FRONTEND_PID_FILE" ]; then
    print_warning "An existing installation directory, service file, or PID file seems to exist."
    if ask_yes_no "Do you want to stop services and remove the existing installation/logs before proceeding?" "N"; then
        PERFORM_CLEANUP=true
    else
        print_info "Proceeding without cleanup. Files might be overwritten or processes might conflict."
    fi
else
    print_info "No existing installation found."
fi


if [ "$PERFORM_CLEANUP" = true ]; then
    print_step "Performing Cleanup"
    print_info "Attempting to stop running backend/frontend processes..."
    # ... (cleanup logic remains the same) ...
    if [ -f "$BACKEND_PID_FILE" ]; then B_PID=$(cat "$BACKEND_PID_FILE"); print_info "Found backend PID file ($BACKEND_PID_FILE) with PID: $B_PID"; if ps -p "$B_PID" > /dev/null; then print_info "Attempting to kill backend process (PID: $B_PID)..."; kill "$B_PID"; sleep 1; if ps -p "$B_PID" > /dev/null; then print_warning "Backend process $B_PID still running, attempting force kill (kill -9)..."; kill -9 "$B_PID" || print_warning "Failed to force kill backend $B_PID."; else print_info "Backend process $B_PID terminated successfully."; fi; else print_info "Backend process $B_PID not found running."; fi; print_info "Removing backend PID file..."; rm -f "$BACKEND_PID_FILE"; else print_info "Backend PID file ($BACKEND_PID_FILE) not found."; fi
    if [ -f "$FRONTEND_PID_FILE" ]; then F_PID=$(cat "$FRONTEND_PID_FILE"); print_info "Found frontend PID file ($FRONTEND_PID_FILE) with PID: $F_PID"; if ps -p "$F_PID" > /dev/null; then print_info "Attempting to kill frontend process (PID: $F_PID)..."; kill "$F_PID"; sleep 1; if ps -p "$F_PID" > /dev/null; then print_warning "Frontend process $F_PID still running, attempting force kill (kill -9)..."; kill -9 "$F_PID" || print_warning "Failed to force kill frontend $F_PID."; else print_info "Frontend process $F_PID terminated successfully."; fi; else print_info "Frontend process $F_PID not found running."; fi; print_info "Removing frontend PID file..."; rm -f "$FRONTEND_PID_FILE"; else print_info "Frontend PID file ($FRONTEND_PID_FILE) not found."; fi
    print_info "Stopping service ${SERVICE_NAME}..."; systemctl stop "$SERVICE_NAME"; print_info "Disabling service ${SERVICE_NAME}..."; systemctl disable "$SERVICE_NAME"; SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"; if [ -f "$SERVICE_FILE_PATH" ]; then print_info "Removing service file ${SERVICE_FILE_PATH}..."; rm -f "$SERVICE_FILE_PATH" || print_warning "Failed to remove service file."; fi; print_info "Reloading systemd daemon..."; systemctl daemon-reload || print_warning "Failed to reload systemd daemon."
    print_info "Removing installation directory ${VLLM_HOME}..."; rm -rf "$VLLM_HOME" || print_warning "Failed to remove installation directory."; print_info "Removing log directory ${LOG_DIR}..."; rm -rf "$LOG_DIR" || print_warning "Failed to remove log directory."
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
chown -R "${VLLM_USER}:${VLLM_GROUP}" "$VLLM_HOME" || { print_error "Failed to change ownership of $VLLM_HOME"; exit 1; }
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
if [ $CD_EXIT_CODE -ne 0 ]; then print_error "Failed to change directory to $VLLM_HOME (Exit code: $CD_EXIT_CODE)"; exit 1; fi
print_info "Successfully changed working directory to $(pwd)"


# --- Setup Python Virtual Environments ---
print_step "Setting up Python Virtual Environments (in $(pwd))"
if [ ! -d "venv" ]; then python3 -m venv venv || { print_error "Failed to create backend venv."; exit 1; }; print_info "Backend venv created."; else print_info "Backend venv exists."; fi
if [ ! -d "frontend_venv" ]; then python3 -m venv frontend_venv || { print_error "Failed to create frontend venv."; exit 1; }; print_info "Frontend venv created."; else print_info "Frontend venv exists."; fi

# --- Install Python Dependencies ---
print_step "Installing Python Dependencies"
# Backend
print_info "Installing backend dependencies..."
"${VLLM_HOME}/venv/bin/pip" install --upgrade pip || print_warning "Failed to upgrade pip."
if [ ! -f "backend/requirements.txt" ]; then
    print_warning "backend/requirements.txt not found. Installing core list."
    "${VLLM_HOME}/venv/bin/pip" install "fastapi[all]" uvicorn huggingface_hub requests pynvml || { print_error "Failed to install core backend dependencies."; exit 1; }
else
    "${VLLM_HOME}/venv/bin/pip" install -r backend/requirements.txt || { print_error "Failed to install backend requirements from file."; exit 1; }
fi
print_success "Backend dependencies installed."
# Frontend
print_info "Installing frontend dependencies..."
"${VLLM_HOME}/frontend_venv/bin/pip" install --upgrade pip || print_warning "Failed to upgrade pip."
if [ ! -f "frontend/requirements.txt" ]; then
     print_warning "frontend/requirements.txt not found. Installing core list."
     "${VLLM_HOME}/frontend_venv/bin/pip" install Flask requests || { print_error "Failed to install core frontend dependencies."; exit 1; }
else
    "${VLLM_HOME}/frontend_venv/bin/pip" install -r frontend/requirements.txt || { print_error "Failed to install frontend requirements from file."; exit 1; }
fi
print_success "Frontend dependencies installed."

# --- Create Initial Config / Active Model Files ---
print_step "Initializing Configuration Files"
CONFIG_DIR_ABS="${VLLM_HOME}/config"
MODEL_CONFIG_PATH_ABS="${CONFIG_DIR_ABS}/model_config.json"
ACTIVE_MODEL_FILE_ABS="${CONFIG_DIR_ABS}/active_model.txt"

mkdir -p "$CONFIG_DIR_ABS" || print_warning "Could not create config directory ${CONFIG_DIR_ABS}"
chown "${VLLM_USER}:${VLLM_GROUP}" "$CONFIG_DIR_ABS" || print_warning "Could not set ownership of ${CONFIG_DIR_ABS}"

# Create empty model config if it doesn't exist
if [ ! -f "$MODEL_CONFIG_PATH_ABS" ]; then
    print_info "Creating empty model config file: ${MODEL_CONFIG_PATH_ABS}"
    echo "{}" > "$MODEL_CONFIG_PATH_ABS"
    chown "${VLLM_USER}:${VLLM_GROUP}" "$MODEL_CONFIG_PATH_ABS"
fi

# Create active model file, try setting first model from config as default
if [ ! -f "$ACTIVE_MODEL_FILE_ABS" ]; then
    print_info "Attempting to set initial active model..."
    # Use jq to get the first key from the model config JSON
    FIRST_MODEL_KEY=$(jq -r 'keys | .[0] // empty' "$MODEL_CONFIG_PATH_ABS")

    if [ -n "$FIRST_MODEL_KEY" ] && [ "$FIRST_MODEL_KEY" != "null" ]; then
        print_info "Setting first model from config ('${FIRST_MODEL_KEY}') as active in ${ACTIVE_MODEL_FILE_ABS}"
        echo "$FIRST_MODEL_KEY" > "$ACTIVE_MODEL_FILE_ABS"
        chown "${VLLM_USER}:${VLLM_GROUP}" "$ACTIVE_MODEL_FILE_ABS"
    else
        print_warning "No models found in ${MODEL_CONFIG_PATH_ABS}. Creating empty active model file."
        print_warning "You will need to add a model and activate it via the API/UI before starting the vLLM service."
        touch "$ACTIVE_MODEL_FILE_ABS"
        chown "${VLLM_USER}:${VLLM_GROUP}" "$ACTIVE_MODEL_FILE_ABS"
    fi
else
    print_info "Active model file already exists: ${ACTIVE_MODEL_FILE_ABS}"
fi


# --- Setup Systemd Service ---
print_step "Setting up Systemd Service"
SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
TEMPLATE_PATH="${VLLM_HOME}/backend/vllm.service.template"

print_info "Checking for template at: ${TEMPLATE_PATH}"
if [ ! -f "$TEMPLATE_PATH" ]; then print_error "Systemd template NOT FOUND at ${TEMPLATE_PATH}"; ls -la "${VLLM_HOME}/backend"; exit 1; else print_success "Systemd template found."; fi

print_info "Creating systemd service file at ${SERVICE_FILE_PATH}"
EFFECTIVE_USER=$VLLM_USER
cat "$TEMPLATE_PATH" | sed -e "s|%USER%|${EFFECTIVE_USER}|g" -e "s|%VLLM_HOME%|${VLLM_HOME}|g" | tee "$SERVICE_FILE_PATH" > /dev/null || { print_error "Failed to create systemd service file."; exit 1; }

print_info "Reloading systemd daemon..."
systemctl daemon-reload || { print_error "Failed to reload systemd daemon."; exit 1; }

print_success "Systemd service configured."
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

# Start Backend
print_info "Starting FastAPI backend on port ${BACKEND_PORT}..."
export VLLM_LOG_DIR="${LOG_DIR}"
nohup "${VLLM_HOME}/venv/bin/python" -m uvicorn backend.main:app --host 0.0.0.0 --port ${BACKEND_PORT} --forwarded-allow-ips '*' > "${LOG_DIR}/backend_stdout.log" 2>&1 &
BACKEND_PID=$!
if [ -d "$VLLM_HOME" ]; then echo $BACKEND_PID > "$BACKEND_PID_FILE"; else print_warning "Cannot write backend PID file, VLLM_HOME not found."; fi
unset VLLM_LOG_DIR
sleep 2
if ps -p $BACKEND_PID > /dev/null; then
   print_success "Backend started successfully (PID: $BACKEND_PID). Logs: ${LOG_DIR}/vllm_backend.log"
else
   print_error "Backend failed to start. Check logs: ${LOG_DIR}/backend_stdout.log and ${LOG_DIR}/vllm_backend.log"
fi

# Start Frontend
print_info "Starting Flask frontend on port ${FRONTEND_PORT}..."
export VLLM_LOG_DIR="${LOG_DIR}"
export FLASK_APP=frontend/app.py
nohup "${VLLM_HOME}/frontend_venv/bin/python" -m flask run --host 0.0.0.0 --port ${FRONTEND_PORT} > "${LOG_DIR}/frontend_stdout.log" 2>&1 &
FRONTEND_PID=$!
if [ -d "$VLLM_HOME" ]; then echo $FRONTEND_PID > "$FRONTEND_PID_FILE"; else print_warning "Cannot write frontend PID file, VLLM_HOME not found."; fi
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
if [[ $EUID -ne 0 ]]; then echo -e "Remember to configure passwordless sudo for service control if you haven't already."; fi
echo -e "Use 'systemctl status ${SERVICE_NAME}' to check the vLLM service status."
echo -e "If the vLLM service needs starting, use the Admin UI or 'systemctl start ${SERVICE_NAME}'."

exit 0