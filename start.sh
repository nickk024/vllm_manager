#!/bin/bash

# start.sh - Setup script for vLLM Backend & Frontend (Multi-Model Serve)

# --- Configuration ---
VLLM_USER="root" # Running as root
VLLM_GROUP="root" # Running as root
VLLM_HOME_DEFAULT="/opt/vllm" # Default installation directory
VLLM_HOME="${VLLM_HOME:-$VLLM_HOME_DEFAULT}" # Allow overriding via environment variable
BACKEND_PORT=8080
FRONTEND_PORT=5000
SERVICE_NAME="vllm" # Systemd service name
BACKEND_PID_FILE="${VLLM_HOME}/backend.pid" # PID file location
FRONTEND_PID_FILE="${VLLM_HOME}/frontend.pid" # PID file location

# Get the directory where this script is located (project root)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/logs" # Log directory in project root
UNIFIED_LOG_FILE="${LOG_DIR}/vllm_manager_app.log" # Unified log file name (matches Python config)

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

# Check for NVIDIA drivers and attempt installation if missing
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA drivers already detected."
    nvidia-smi -L
else
    print_warning "NVIDIA drivers (nvidia-smi) not found. Attempting installation..."

    # --- Distro Check and Sources Modification (Debian specific) ---
    if [ -f /etc/debian_version ]; then
        print_info "Debian-based system detected. Checking APT sources for non-free components..."
        SOURCES_FILE="/etc/apt/sources.list"
        SOURCES_BACKUP="${SOURCES_FILE}.bak_$(date +%Y%m%d%H%M%S)"
        MODIFIED_SOURCES=false

        # Check if non-free components are already enabled for the main repo lines containing 'main'
        # We check if *any* relevant line is missing *all* required components
        if grep -qE '^deb\s+.*deb\.debian\.org/debian.*\smain' "$SOURCES_FILE" && \
           ! grep -qE '^deb\s+.*deb\.debian\.org/debian.*\smain\s+.*\bcontrib\b.*\bnon-free\b.*\bnon-free-firmware\b' "$SOURCES_FILE"; then

            print_warning "Main Debian repository might be missing contrib/non-free/non-free-firmware components needed for NVIDIA drivers."
            print_info "Backing up $SOURCES_FILE to $SOURCES_BACKUP..."
            # Use sudo for cp/sed as we are root but file permissions might be strict
            cp "$SOURCES_FILE" "$SOURCES_BACKUP" || { print_error "Failed to backup sources.list. Aborting modification."; exit 1; }

            print_info "Attempting to add contrib non-free non-free-firmware to main Debian repo lines..."
            # Add components after 'main', preserving anything that might already be there but not all three.
            # This handles cases like 'main', 'main contrib', 'main non-free', etc.
            sed -i -E 's/^(deb\s+.*deb\.debian\.org\/debian.*\smain)(\s*[^#]*)$/\1 contrib non-free non-free-firmware\2/' "$SOURCES_FILE"

            # Verify if modification seemed successful by checking again
            if ! grep -qE '^deb\s+.*deb\.debian\.org/debian.*\smain\s+.*\bcontrib\b.*\bnon-free\b.*\bnon-free-firmware\b' "$SOURCES_FILE"; then
                 print_warning "Automatic modification of sources.list might have failed or components were already partially present in an unexpected format. Please check $SOURCES_FILE manually."
                 # Don't set MODIFIED_SOURCES=true if the final check fails
            else
                 print_success "Added/Ensured contrib non-free non-free-firmware components in sources.list."
                 MODIFIED_SOURCES=true
            fi
        else
            print_info "Contrib/non-free/non-free-firmware components seem already enabled or main repo line not found/unexpected format."
        fi

        # Run apt update if sources were modified
        if [ "$MODIFIED_SOURCES" = true ]; then
            print_info "Running apt-get update after modifying sources..."
            apt-get update -q || { print_error "apt-get update failed after modifying sources."; exit 1; }
        else
             # Run update anyway before installing packages
             print_info "Running apt-get update..."
             apt-get update -q || print_warning "apt-get update failed, continuing anyway..."
        fi
    else
        print_warning "Non-Debian system detected. Skipping automatic APT source modification for NVIDIA drivers."
        # Run update before attempting install
        apt-get update -q || print_warning "apt-get update failed, continuing anyway..."
    fi
    # --- End Distro Check ---

    # Install nvidia-detect first
    print_info "Installing nvidia-detect..."
    apt-get install -y -q nvidia-detect || print_warning "Failed to install nvidia-detect. Will try fallback driver."

    DRIVER_PACKAGE=""
    if command -v nvidia-detect &> /dev/null; then
        DETECT_OUTPUT=$(nvidia-detect 2>/dev/null)
        # Extract recommended package (handles nvidia-driver and nvidia-driver-XXX)
        DRIVER_PACKAGE=$(echo "$DETECT_OUTPUT" | grep -oP '^\s*\Knvidia-driver(?:-\d+)?')
        if [ -n "$DRIVER_PACKAGE" ]; then
            print_info "nvidia-detect recommended package: $DRIVER_PACKAGE"
        else
            print_warning "nvidia-detect ran but did not recommend a specific driver package. Falling back to 'nvidia-driver'."
            DRIVER_PACKAGE="nvidia-driver" # Fallback for Debian
        fi
    else
        print_warning "nvidia-detect not found. Falling back to installing 'nvidia-driver' metapackage."
        DRIVER_PACKAGE="nvidia-driver" # Fallback for Debian
    fi

    if [ -n "$DRIVER_PACKAGE" ]; then
        print_info "Attempting to install $DRIVER_PACKAGE..."
        # Ensure non-free components are enabled (often needed) - best effort
        # This might require manual source list editing beforehand if not default
        print_info "Running apt-get update again to ensure lists are fresh..." # Add extra update
        apt-get update -q || print_warning "Second apt-get update failed, install might still fail..."
        apt-get install -y -q "$DRIVER_PACKAGE"
        INSTALL_EXIT_CODE=$?
        if [ $INSTALL_EXIT_CODE -ne 0 ]; then
            print_error "Failed to install $DRIVER_PACKAGE (Exit code: $INSTALL_EXIT_CODE). Please install NVIDIA drivers manually and re-run."
            exit 1
        else
            print_success "$DRIVER_PACKAGE installed. Checking nvidia-smi again..."
            # Check again
            if command -v nvidia-smi &> /dev/null; then
                print_success "NVIDIA drivers (nvidia-smi) successfully installed and detected."
                nvidia-smi -L
                print_warning "A system reboot might be required for drivers to function correctly."
            else
                print_error "nvidia-smi not found even after installing $DRIVER_PACKAGE. Manual installation/reboot might be required."
                exit 1
            fi
        fi
    else
         print_error "Could not determine NVIDIA driver package to install. Please install manually."
         exit 1
    fi
fi

if ! command -v python3 &> /dev/null; then print_error "python3 not found."; exit 1; fi
if ! python3 -m venv -h &> /dev/null; then print_error "python3-venv module not found."; exit 1; fi
print_info "Python 3 and venv found."
if ! command -v rsync &> /dev/null; then print_warning "rsync command not found. Will attempt to install."; fi
if ! command -v jq &> /dev/null; then print_warning "jq command not found. Will attempt to install."; fi # jq no longer needed here
print_info "Checks complete."


# --- Optional Cleanup ---
print_step "Checking for Existing Installation"
PERFORM_CLEANUP=false
# Check for install dir OR service file OR PID files
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
    if [ -f "$BACKEND_PID_FILE" ]; then B_PID=$(cat "$BACKEND_PID_FILE"); print_info "Found backend PID file ($BACKEND_PID_FILE) with PID: $B_PID"; if ps -p "$B_PID" > /dev/null; then print_info "Attempting to kill backend process (PID: $B_PID)..."; kill "$B_PID"; sleep 1; if ps -p "$B_PID" > /dev/null; then print_warning "Backend process $B_PID still running, attempting force kill (kill -9)..."; kill -9 "$B_PID" || print_warning "Failed to force kill backend $B_PID."; else print_info "Backend process $B_PID terminated successfully."; fi; else print_info "Backend process $B_PID not found running."; fi; print_info "Removing backend PID file..."; rm -f "$BACKEND_PID_FILE"; else print_info "Backend PID file ($BACKEND_PID_FILE) not found."; fi
    if [ -f "$FRONTEND_PID_FILE" ]; then F_PID=$(cat "$FRONTEND_PID_FILE"); print_info "Found frontend PID file ($FRONTEND_PID_FILE) with PID: $F_PID"; if ps -p "$F_PID" > /dev/null; then print_info "Attempting to kill frontend process (PID: $F_PID)..."; kill "$F_PID"; sleep 1; if ps -p "$F_PID" > /dev/null; then print_warning "Frontend process $F_PID still running, attempting force kill (kill -9)..."; kill -9 "$F_PID" || print_warning "Failed to force kill frontend $F_PID."; else print_info "Frontend process $F_PID terminated successfully."; fi; else print_info "Frontend process $F_PID not found running."; fi; print_info "Removing frontend PID file..."; rm -f "$FRONTEND_PID_FILE"; else print_info "Frontend PID file ($FRONTEND_PID_FILE) not found."; fi
    # Stop Ray cluster if running
    print_info "Attempting to stop Ray cluster..."
    if command -v ray &> /dev/null; then
        "${VLLM_HOME}/venv/bin/ray" stop || print_warning "Failed to stop Ray cluster. It might not have been running."
        print_info "Ray stop command executed."
    else
        print_warning "Ray command not found, cannot stop cluster."
    fi
    # Systemd cleanup (keep for legacy, but it should fail gracefully if service removed)
    print_info "Stopping service ${SERVICE_NAME}..."; systemctl stop "$SERVICE_NAME" || print_warning "Failed to stop systemd service (might not exist)."
    print_info "Disabling service ${SERVICE_NAME}..."; systemctl disable "$SERVICE_NAME" || print_warning "Failed to disable systemd service (might not exist)."
    SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"; if [ -f "$SERVICE_FILE_PATH" ]; then print_info "Removing service file ${SERVICE_FILE_PATH}..."; rm -f "$SERVICE_FILE_PATH" || print_warning "Failed to remove service file."; fi; print_info "Reloading systemd daemon..."; systemctl daemon-reload || print_warning "Failed to reload systemd daemon."
    print_info "Removing installation directory ${VLLM_HOME}..."; rm -rf "$VLLM_HOME" || print_warning "Failed to remove installation directory."
    print_info "Removing log directory ${LOG_DIR}..."; rm -rf "$LOG_DIR" || print_warning "Failed to remove log directory."
    print_success "Cleanup finished."
fi


# --- System Dependencies ---
print_step "Installing System Dependencies"
apt-get update -q || { print_error "apt-get update failed."; exit 1; }
# Removed jq as it's no longer needed by start script
apt-get install -y -q build-essential git wget curl rsync python3-dev python3-venv || { print_error "Failed to install essential packages."; exit 1; }
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

# --- Create Log Directory (Standardized) ---
# Use unified log path from config
LOG_DIR_STD="${VLLM_HOME}/logs"  # Unified logging path from config
UNIFIED_LOG_FILE="${LOG_DIR_STD}/vllm_manager_app.log"  # Unified log file path
print_step "Creating Standard Log Directory: ${LOG_DIR_STD}"
mkdir -p "$LOG_DIR_STD" || { print_error "Failed to create log directory $LOG_DIR_STD"; exit 1; }
chown "${VLLM_USER}:${VLLM_GROUP}" "$LOG_DIR_STD" || print_warning "Could not set ownership of log directory ${LOG_DIR_STD}"

# Create a symlink to the unified log file in the project root for easier access
if [ ! -L "${SCRIPT_DIR}/unified_log.log" ]; then
    print_info "Creating symlink to unified log file in project root"
    ln -sf "${UNIFIED_LOG_FILE}" "${SCRIPT_DIR}/unified_log.log" || print_warning "Could not create symlink to unified log file"
fi

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
# Ensure clean environment
print_info "Removing existing virtual environment and rebuilding..."
rm -rf "${SCRIPT_DIR}/venv"
python3 -m venv "${SCRIPT_DIR}/venv"
source "${SCRIPT_DIR}/venv/bin/activate"
# Backend
print_info "Installing backend dependencies..."
"${VLLM_HOME}/venv/bin/pip" install --upgrade pip || print_warning "Failed to upgrade pip."
if [ ! -f "backend/requirements.txt" ]; then
    print_warning "backend/requirements.txt not found. Installing core list."
    # Try installing with ray[serve,default] to potentially include more components
    print_info "Installing core backend dependencies (using ray[serve,default])..."
    "${VLLM_HOME}/venv/bin/pip" install "fastapi[all]" uvicorn huggingface_hub requests pynvml "ray[serve,default]" vllm pytest httpx || { print_error "Failed to install core backend dependencies (including vllm, pytest, httpx, ray[serve,default])."; exit 1; }
    # Explicit install is likely redundant now, removing it.
    # print_info "Ensuring ray[serve] is installed..."
    # "${VLLM_HOME}/venv/bin/pip" install "ray[serve]" || { print_error "Failed to explicitly install Ray Serve."; exit 1; }
else
    "${VLLM_HOME}/venv/bin/pip" install -r backend/requirements.txt || { print_error "Failed to install backend requirements from file."; exit 1; }
    # Ensure Ray Serve is installed even if requirements.txt is present
    print_info "Ensuring ray[serve,default] is installed (from requirements)..."
    "${VLLM_HOME}/venv/bin/pip" install "ray[serve,default]" || { print_error "Failed to install Ray Serve with default extras."; exit 1; }
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

# --- Create Initial Config Files (if they don't exist) ---
print_step "Initializing Configuration Files"
CONFIG_DIR_ABS="${VLLM_HOME}/config"
MODEL_CONFIG_PATH_ABS="${CONFIG_DIR_ABS}/model_config.json"
# ACTIVE_MODEL_FILE_ABS="${CONFIG_DIR_ABS}/active_model.txt" # No longer needed

mkdir -p "$CONFIG_DIR_ABS" || print_warning "Could not create config directory ${CONFIG_DIR_ABS}"
chown "${VLLM_USER}:${VLLM_GROUP}" "$CONFIG_DIR_ABS" || print_warning "Could not set ownership of ${CONFIG_DIR_ABS}"

if [ ! -f "$MODEL_CONFIG_PATH_ABS" ]; then
    print_info "Creating empty model config file: ${MODEL_CONFIG_PATH_ABS}"
    echo "{}" > "$MODEL_CONFIG_PATH_ABS"
    chown "${VLLM_USER}:${VLLM_GROUP}" "$MODEL_CONFIG_PATH_ABS"
fi
# Removed active_model.txt creation


# --- Start Ray Cluster Head Node ---
print_step "Starting Ray Cluster Head Node"
print_info "Attempting to stop any existing Ray instance..."
"${VLLM_HOME}/venv/bin/ray" stop || print_warning "Failed to stop existing Ray instance (might not be running)."
print_info "Starting Ray head node (required for Ray Serve)..."
"${VLLM_HOME}/venv/bin/ray" start --head --port=6379 || { print_error "Failed to start Ray head node."; exit 1; }
print_success "Ray head node started."

# --- Start Services ---
print_step "Starting Backend and Frontend Services"

# Start Backend
print_info "Starting FastAPI backend on port ${BACKEND_PORT}..."
# Redirect stderr to a file to capture startup errors
BACKEND_STDERR_LOG="/opt/vllm/backend_stderr.log"
print_info "Redirecting backend startup stderr to ${BACKEND_STDERR_LOG}"
# Ensure the log directory exists for the stderr log as well
mkdir -p "$(dirname "$BACKEND_STDERR_LOG")"
chown "${VLLM_USER}:${VLLM_GROUP}" "$(dirname "$BACKEND_STDERR_LOG")" || print_warning "Could not set ownership for backend stderr log directory"

# Start backend, redirecting stderr
nohup "${VLLM_HOME}/venv/bin/python" -m uvicorn backend.main:app --host 0.0.0.0 --port ${BACKEND_PORT} --forwarded-allow-ips '*' > /dev/null 2> "$BACKEND_STDERR_LOG" &
BACKEND_PID=$!
if [ -d "$VLLM_HOME" ]; then echo $BACKEND_PID > "$BACKEND_PID_FILE"; else print_warning "Cannot write backend PID file, VLLM_HOME not found."; fi
sleep 2
if ps -p $BACKEND_PID > /dev/null; then
   print_success "Backend started successfully (PID: $BACKEND_PID). Logs: ${UNIFIED_LOG_FILE}"
else
  print_error "Backend failed to start. Check logs: ${UNIFIED_LOG_FILE} and ${BACKEND_STDERR_LOG}" # Mention stderr log too
fi

# Start Frontend
print_info "Starting Flask frontend on port ${FRONTEND_PORT}..."
# Set the backend URL environment variable for the frontend
export VLLM_BACKEND_URL="http://localhost:${BACKEND_PORT}/api/v1/manage"
print_info "Setting VLLM_BACKEND_URL=${VLLM_BACKEND_URL}"
export FLASK_APP=frontend/app.py
nohup "${VLLM_HOME}/frontend_venv/bin/python" -m flask run --host 0.0.0.0 --port ${FRONTEND_PORT} > /dev/null 2>&1 &
FRONTEND_PID=$!
if [ -d "$VLLM_HOME" ]; then echo $FRONTEND_PID > "$FRONTEND_PID_FILE"; else print_warning "Cannot write frontend PID file, VLLM_HOME not found."; fi
sleep 1
if ps -p $FRONTEND_PID > /dev/null; then
   print_success "Frontend started successfully (PID: $FRONTEND_PID). Logs: ${UNIFIED_LOG_FILE}"
else
   print_error "Frontend failed to start. Check logs: ${UNIFIED_LOG_FILE}"
fi

# Ray Serve deployments will be launched by the backend (FastAPI) on startup.
print_step "Setup Complete!"
echo -e "Backend API should be running at: ${BOLD}http://<server_ip>:${BACKEND_PORT}${NC}"
echo -e "Frontend Admin UI should be running at: ${BOLD}http://<server_ip>:${FRONTEND_PORT}${NC}"
echo -e "Ray Serve is now used for model serving. The OpenAI-compatible endpoint will be available on port 8000 (Ray Serve)."
echo -e "Use the Admin UI to manage models and monitor Ray Serve deployments."
echo -e "Configure models via UI/API, and Ray Serve will handle dynamic loading/unloading as needed."

exit 0