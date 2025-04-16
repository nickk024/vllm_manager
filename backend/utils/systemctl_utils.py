import subprocess
import logging
from fastapi import HTTPException, status as http_status
from ..models import ServiceActionResponse # Relative import for Pydantic model
from ..config import SERVICE_NAME # Relative import for service name constant

logger = logging.getLogger(__name__)

# WARNING: Running sudo commands from a web server is a security risk.
# Ensure the user running the FastAPI server has *specific, limited* passwordless sudo rights
# ONLY for 'systemctl start/stop/restart/enable/disable vllm'.
# Consider alternatives like a message queue or a dedicated privileged helper process
# for production environments.

def run_systemctl_command(action: str) -> ServiceActionResponse:
    """
    Helper function to run systemctl commands with sudo for the configured service.

    Args:
        action: The systemctl action (e.g., 'start', 'stop', 'restart', 'enable', 'disable').

    Returns:
        ServiceActionResponse indicating success or failure.

    Raises:
        HTTPException: If the command fails or prerequisites (sudo/systemctl) are missing.
    """
    # Validate action? Maybe not necessary if only called internally.
    valid_actions = ["start", "stop", "restart", "enable", "disable", "is-active", "is-enabled"]
    if action not in valid_actions:
         logger.error(f"Invalid systemctl action requested: {action}")
         # This should ideally not happen if called correctly from routers
         raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid internal action requested.")

    command = ["sudo", "systemctl", action, SERVICE_NAME]
    log_command = " ".join(command) # For logging, don't log sensitive info if command had it

    try:
        logger.info(f"Executing systemctl command: {log_command}")
        # For status checks, we don't want check=True as failure is expected for inactive/disabled
        should_check = action not in ["is-active", "is-enabled"]
        result = subprocess.run(command, check=should_check, capture_output=True, text=True, timeout=15) # Added timeout

        if result.returncode == 0:
             logger.info(f"Command '{log_command}' successful. Output: {result.stdout.strip()} {result.stderr.strip()}")
             # For status checks, return the actual output
             if action in ["is-active", "is-enabled"]:
                  return ServiceActionResponse(status="ok", message=result.stdout.strip())
             else:
                  return ServiceActionResponse(status="ok", message=f"Service '{SERVICE_NAME}' {action} successful.")
        else:
             # Handle non-zero exit codes for status checks specifically
             if action in ["is-active", "is-enabled"]:
                  logger.info(f"Command '{log_command}' returned status: {result.stdout.strip()}")
                  return ServiceActionResponse(status="ok", message=result.stdout.strip()) # e.g., "inactive", "disabled"
             else:
                  # This part should technically be caught by CalledProcessError if check=True
                  logger.error(f"Command '{log_command}' failed with exit code {result.returncode}. Stderr: {result.stderr.strip()}")
                  raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                                      detail=f"Failed to {action} service '{SERVICE_NAME}'. Error: {result.stderr.strip()}")

    except FileNotFoundError:
        logger.error(f"Error: 'sudo' or 'systemctl' command not found when trying to run '{log_command}'.")
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Server configuration error: cannot find sudo or systemctl.")
    except subprocess.CalledProcessError as e:
        # This catches errors when check=True (i.e., for start, stop, restart, enable, disable)
        logger.error(f"Failed to {action} {SERVICE_NAME} service via '{log_command}'. Stderr: {e.stderr.strip()}")
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to {action} service '{SERVICE_NAME}'. Error: {e.stderr.strip()}")
    except subprocess.TimeoutExpired:
         logger.error(f"Command '{log_command}' timed out after 15 seconds.")
         raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Command '{action} {SERVICE_NAME}' timed out.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during systemctl {action} via '{log_command}': {e}")
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected error occurred during service {action}.")