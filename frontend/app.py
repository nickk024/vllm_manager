import os
import sys # For stderr
import requests
import logging # Import logging
import logging.handlers # For file logging
from flask import Flask, render_template, request, redirect, url_for, flash, Response

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages

# --- Logging Configuration ---
# Get log directory from environment variable set by start.sh
# Fallback to a 'logs' directory relative to this script's location if env var not set
LOG_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "logs") # Default relative to project root
LOG_DIR = os.environ.get("VLLM_LOG_DIR", LOG_DIR_DEFAULT)
LOG_FILE = os.path.join(LOG_DIR, "vllm_frontend.log")
LOG_LEVEL = logging.DEBUG # Set log level to DEBUG for file

# Ensure log directory exists
log_dir_valid = False
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_dir_valid = True
    print(f"[Flask INFO] Logging directory set to: {LOG_DIR}") # Print confirmation at startup
except OSError as e:
     print(f"[Flask ERROR] Could not create or access log directory: {LOG_DIR}. Error: {e}", file=sys.stderr)
     # Frontend might still run but won't log to file

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure Flask's built-in logger
app.logger.setLevel(LOG_LEVEL) # Set overall level for Flask logger

# Clear default handlers? Optional, but can prevent duplicate console logs if Flask adds one by default
# It's generally safer to leave Flask's default console handler unless it causes issues.
# We will ensure the console handler level is set correctly below.
for handler in app.logger.handlers[:]:
     # Remove default handler if we are adding our own configured one
     # Or just find it and set its level
     app.logger.removeHandler(handler)


# Add console handler (Flask usually adds one, but let's be explicit for clarity and level setting)
console_handler = logging.StreamHandler(sys.stderr) # Log to stderr
console_handler.setLevel(logging.INFO) # Keep console INFO
console_handler.setFormatter(formatter)
app.logger.addHandler(console_handler)


# Create rotating file handler only if log dir is valid
if log_dir_valid:
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=3 # Smaller size for frontend logs
        )
        file_handler.setLevel(LOG_LEVEL) # Set file handler level
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler) # Add file handler to Flask's logger
        app.logger.info(f"--- Flask Logging configured (Console: INFO, File: {LOG_LEVEL} at {LOG_FILE}) ---")
    except Exception as e:
         print(f"[Flask ERROR] Failed to create file log handler for {LOG_FILE}. Error: {e}", file=sys.stderr)
         app.logger.error(f"Failed to create file log handler for {LOG_FILE}. Error: {e}") # Log error to console
else:
     app.logger.warning("--- Flask Logging configured (Console only) ---")

# --- End Logging Configuration ---


# Configuration for the backend API URL
BACKEND_API_URL = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8080/api/v1")

def call_backend(method: str, endpoint: str, json_data=None) -> dict | None:
    """Helper function to call the backend API."""
    url = f"{BACKEND_API_URL}{endpoint}"
    action_description = f"{method} {endpoint}"
    try:
        app.logger.info(f"Calling backend: {action_description} with data: {json_data}")
        response = requests.request(method, url, json=json_data, timeout=30)
        response.raise_for_status()

        if response.status_code == 204:
             app.logger.info(f"Backend call successful ({action_description}): Status 204 No Content")
             return {"status": "ok", "message": "Action successful (no content)."}
        try:
            response_json = response.json()
            app.logger.info(f"Backend call successful ({action_description}): {response_json}")
            if 'status' not in response_json or 'message' not in response_json:
                 return {"status": "ok", "message": "Received successful but non-standard response.", "details": response_json}
            return response_json
        except requests.exceptions.JSONDecodeError:
             app.logger.warning(f"Backend call successful ({action_description}) but response was not JSON: {response.text[:100]}...")
             return {"status": "ok", "message": "Action successful (non-JSON response received).", "details": response.text}

    except requests.exceptions.HTTPError as e:
        error_details = f"HTTP Error: {e.response.status_code}"
        try:
            backend_error = e.response.json()
            error_msg = backend_error.get('detail', backend_error.get('message', e.response.text))
            error_details += f" - {error_msg}"
        except requests.exceptions.JSONDecodeError:
             error_details += f" - {e.response.text}"
        app.logger.error(f"Error calling backend ({action_description}): {error_details}")
        flash(f"Error performing action ({action_description}): {error_details}", "error")
        return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to backend ({action_description}): {e}")
        flash(f"Error connecting to backend ({url}): {e}", "error")
        return None
    except Exception as e:
        app.logger.error(f"Unexpected error during backend call ({action_description}): {e}")
        flash(f"An unexpected error occurred: {e}", "error")
        return None


@app.route('/')
def index():
    """Dashboard showing status and models."""
    app.logger.info("Accessing index route '/'")
    status_data = call_backend("GET", "/service/status")
    monitoring_data = call_backend("GET", "/monitoring/stats")

    service_status = "Error Fetching"; service_enabled = "Error Fetching"
    active_model_key = "Error Fetching"; models = []; stats = None

    if status_data:
        service_status = status_data.get("service_status", "Unknown")
        service_enabled = status_data.get("service_enabled", "Unknown")
        active_model_key = status_data.get("active_model_key", "N/A")
        models = status_data.get("configured_models", [])
    else:
         app.logger.warning("Failed to fetch service status from backend.")

    if monitoring_data:
        stats = monitoring_data
    else:
         app.logger.warning("Failed to fetch monitoring stats from backend.")


    return render_template('index.html',
                           service_status=service_status,
                           service_enabled=service_enabled,
                           active_model_key=active_model_key,
                           models=models,
                           monitoring_stats=stats,
                           config_path="model_config.json"
                           )

@app.route('/service/<action>', methods=['POST'])
def handle_service_action(action: str):
    """Handles start, stop, enable, disable actions."""
    app.logger.info(f"Handling service action request: {action}")
    valid_actions = ["start", "stop", "enable", "disable"]
    if action not in valid_actions:
        flash(f"Invalid service action requested: {action}", "error")
        return redirect(url_for('index'))

    result = call_backend("POST", f"/service/{action}")

    if result and result.get("status") == "ok":
        flash(f"Service action '{action}' initiated successfully: {result.get('message', '')}", "success")

    return redirect(url_for('index'))

@app.route('/activate/<model_key>', methods=['POST'])
def handle_activate_model(model_key: str):
    """Handles activation request for a specific model."""
    app.logger.info(f"Handling activation request for model: {model_key}")
    if not model_key:
         flash("No model key provided for activation.", "error")
         return redirect(url_for('index'))

    result = call_backend("POST", f"/service/activate/{model_key}")

    if result and result.get("status") == "ok":
        flash(f"Activation for model '{model_key}' initiated successfully. {result.get('message', '')}", "success")

    return redirect(url_for('index'))


@app.route('/download/<model_name>', methods=['POST'])
def handle_download_model(model_name: str):
    """Handles download request for a specific model."""
    app.logger.info(f"Handling download request for model: {model_name}")
    if not model_name:
         flash("No model name provided for download.", "error")
         return redirect(url_for('index'))

    hf_token = request.form.get('hf_token')
    force_download = request.form.get('force') == 'on' # Check if force checkbox exists and is checked
    payload = {
        "models": [model_name],
        "token": hf_token if hf_token else None,
        "force": force_download
    }
    result = call_backend("POST", "/models/download", json_data=payload)

    if result and result.get("status") == "ok":
        flash(f"Download task for model '{model_name}' started in background.", "success")

    return redirect(url_for('index'))


if __name__ == '__main__':
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        if not os.path.exists(os.path.join(template_dir, 'index.html')):
             with open(os.path.join(template_dir, 'index.html'), 'w') as f:
                  f.write('<html><body><h1>Admin UI Placeholder</h1><p>Template not fully generated yet.</p></body></html>')

    app.logger.info(f"Flask frontend starting...")
    app.logger.info(f"Connecting to backend API at: {BACKEND_API_URL}")
    # Set debug=False for production environments
    app.run(host='0.0.0.0', port=5000, debug=True) # Keep debug=True for development ease