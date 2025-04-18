#!/bin/bash

# uninstall.sh - Removes the vLLM Manager installation created by start.sh

# --- Configuration ---
VLLM_USER="root" # User assumed by start.sh
VLLM_GROUP="root" # Group assumed by start.sh
VLLM_HOME_DEFAULT="/opt/vllm" # Default installation directory used by start.sh
VLLM_HOME="${VLLM_HOME:-$VLLM_HOME_DEFAULT}" # Allow overriding via environment variable
SERVICE_NAME="vllm" # Systemd service name potentially used
BACKEND_PID_FILE="${VLLM_HOME}/backend.pid" # PID file location
FRONTEND_PID_FILE="${VLLM_HOME}/frontend.pid" # PID file location

# Get the directory where this script is located (project root)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_SYMLINK="${SCRIPT_DIR}/unified_log.log"

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
if [[ $EUID -ne 0 ]]; then print_error "This script needs root privileges to remove $VLLM_HOME and manage services."; exit 1; fi

if [ ! -d "$VLLM_HOME" ] && [ ! -f "$BACKEND_PID_FILE" ] && [ ! -f "$FRONTEND_PID_FILE" ] && [ ! -L "$LOG_SYMLINK" ]; then
    print_warning "No installation directory ($VLLM_HOME), PID files, or log symlink found. Nothing to uninstall."
    # Check for service file just in case
    SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
    if [ -f "$SERVICE_FILE_PATH" ]; then
        print_warning "However, a systemd service file was found at ${SERVICE_FILE_PATH}. Will attempt to remove it."
    else
        exit 0
    fi
fi

# --- Stop Running Processes ---
print_step "Stopping Running Processes"

# Stop Backend
if [ -f "$BACKEND_PID_FILE" ]; then
    B_PID=$(cat "$BACKEND_PID_FILE")
    print_info "Found backend PID file ($BACKEND_PID_FILE) with PID: $B_PID"
    if ps -p "$B_PID" > /dev/null; then
        print_info "Attempting to kill backend process (PID: $B_PID)..."
        kill "$B_PID"
        sleep 1
        if ps -p "$B_PID" > /dev/null; then
            print_warning "Backend process $B_PID still running, attempting force kill (kill -9)..."
            kill -9 "$B_PID" || print_warning "Failed to force kill backend $B_PID."
        else
            print_info "Backend process $B_PID terminated successfully."
        fi
    else
        print_info "Backend process $B_PID not found running."
    fi
    print_info "Removing backend PID file..."
    rm -f "$BACKEND_PID_FILE"
else
    print_info "Backend PID file ($BACKEND_PID_FILE) not found."
fi

# Stop Frontend
if [ -f "$FRONTEND_PID_FILE" ]; then
    F_PID=$(cat "$FRONTEND_PID_FILE")
    print_info "Found frontend PID file ($FRONTEND_PID_FILE) with PID: $F_PID"
    if ps -p "$F_PID" > /dev/null; then
        print_info "Attempting to kill frontend process (PID: $F_PID)..."
        kill "$F_PID"
        sleep 1
        if ps -p "$F_PID" > /dev/null; then
            print_warning "Frontend process $F_PID still running, attempting force kill (kill -9)..."
            kill -9 "$F_PID" || print_warning "Failed to force kill frontend $F_PID."
        else
            print_info "Frontend process $F_PID terminated successfully."
        fi
    else
        print_info "Frontend process $F_PID not found running."
    fi
    print_info "Removing frontend PID file..."
    rm -f "$FRONTEND_PID_FILE"
else
    print_info "Frontend PID file ($FRONTEND_PID_FILE) not found."
fi

# Stop Ray cluster if running (use venv path if it exists)
print_info "Attempting to stop Ray cluster..."
RAY_CMD="${VLLM_HOME}/venv/bin/ray"
if [ -f "$RAY_CMD" ]; then
    "$RAY_CMD" stop || print_warning "Failed to stop Ray cluster. It might not have been running or venv is incomplete."
    print_info "Ray stop command executed."
else
    print_warning "Ray command not found at ${RAY_CMD}, cannot stop cluster via script. Check if Ray is running manually."
fi

# --- Remove Systemd Service (Optional, based on start.sh cleanup) ---
print_step "Removing Systemd Service (if exists)"
SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
if [ -f "$SERVICE_FILE_PATH" ]; then
    print_info "Stopping service ${SERVICE_NAME}..."
    systemctl stop "$SERVICE_NAME" || print_warning "Failed to stop systemd service (might not be running)."
    print_info "Disabling service ${SERVICE_NAME}..."
    systemctl disable "$SERVICE_NAME" || print_warning "Failed to disable systemd service."
    print_info "Removing service file ${SERVICE_FILE_PATH}..."
    rm -f "$SERVICE_FILE_PATH" || print_warning "Failed to remove service file."
    print_info "Reloading systemd daemon..."
    systemctl daemon-reload || print_warning "Failed to reload systemd daemon."
    print_success "Systemd service cleanup attempted."
else
    print_info "Systemd service file ${SERVICE_FILE_PATH} not found. Skipping service removal."
fi

# --- Remove Installation Files ---
print_step "Removing Installation Directory"
if [ -d "$VLLM_HOME" ]; then
    print_info "Removing directory ${VLLM_HOME}..."
    rm -rf "$VLLM_HOME"
    if [ $? -eq 0 ]; then
        print_success "Installation directory removed."
    else
        print_error "Failed to remove installation directory ${VLLM_HOME}."
    fi
else
    print_info "Installation directory ${VLLM_HOME} not found."
fi

# --- Remove Log Symlink ---
print_step "Removing Log Symlink"
if [ -L "$LOG_SYMLINK" ]; then
    print_info "Removing log symlink ${LOG_SYMLINK}..."
    rm -f "$LOG_SYMLINK"
    if [ $? -eq 0 ]; then
        print_success "Log symlink removed."
    else
        print_error "Failed to remove log symlink ${LOG_SYMLINK}."
    fi
else
     print_info "Log symlink ${LOG_SYMLINK} not found."
fi


print_step "Uninstall Complete!"
exit 0