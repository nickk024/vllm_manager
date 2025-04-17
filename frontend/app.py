import os
import sys
import requests
import logging
import logging.handlers
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from urllib.parse import quote # For URL encoding model IDs

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Logging Configuration ---
# Standardized log directory and file name
LOG_DIR_STD = "/opt/vllm/logs"
# Use the same log file as the backend for unified logging
UNIFIED_LOG_FILE = os.path.join(LOG_DIR_STD, "vllm_manager_app.log")
LOG_LEVEL = logging.DEBUG

# Ensure the standard log directory exists (start.sh should create it, but double-check)
try:
    os.makedirs(LOG_DIR_STD, exist_ok=True)
    print(f"[Flask INFO] Standard log directory set to: {LOG_DIR_STD}")
except OSError as e:
     print(f"[Flask ERROR] Could not create/access standard log directory: {LOG_DIR_STD}. Error: {e}", file=sys.stderr)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [Flask] %(message)s')
app.logger.setLevel(LOG_LEVEL)
for handler in app.logger.handlers[:]: app.logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
app.logger.addHandler(console_handler)

try:
    # Use the unified log file path
    file_handler = logging.handlers.RotatingFileHandler(UNIFIED_LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.info(f"--- Flask Logging configured (Console: INFO, File: {LOG_LEVEL} at {UNIFIED_LOG_FILE}) ---")
except Exception as e:
     print(f"[Flask ERROR] Failed to create file log handler for {FRONTEND_LOG_FILE}. Error: {e}", file=sys.stderr)
     app.logger.error(f"Failed to create file log handler for {FRONTEND_LOG_FILE}. Error: {e}")
# --- End Logging Configuration ---

BACKEND_API_URL = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8080/api/v1/manage") # Point to management prefix

def call_backend(method: str, endpoint: str, params: dict = None, json_data=None) -> dict | list | None:
    """Helper function to call the backend management API."""
    url = f"{BACKEND_API_URL}{endpoint}" # Endpoint should start with / (e.g., /service/status)
    action_description = f"{method} {url}" # Log full URL
    try:
        app.logger.info(f"Calling backend: {action_description} with params: {params}, data: {json_data}")
        response = requests.request(method, url, params=params, json=json_data, timeout=30)
        response.raise_for_status()
        if response.status_code == 204:
             app.logger.info(f"Backend call successful ({action_description}): Status 204 No Content")
             return {"status": "ok", "message": "Action successful (no content)."}
        try:
            response_json = response.json()
            app.logger.info(f"Backend call successful ({action_description}): {response_json}")
            return response_json # Return raw JSON (list or dict)
        except requests.exceptions.JSONDecodeError:
             app.logger.warning(f"Backend call successful ({action_description}) but response was not JSON: {response.text[:100]}...")
             flash(f"Received non-JSON response from backend for {action_description}", "warning")
             return None
    except requests.exceptions.HTTPError as e:
        error_details = f"HTTP Error: {e.response.status_code}"
        try:
            backend_error = e.response.json()
            error_msg = backend_error.get('detail', backend_error.get('message', e.response.text))
            error_details += f" - {error_msg}"
        except requests.exceptions.JSONDecodeError: error_details += f" - {e.response.text}"
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
    """Dashboard showing status, models, popular models, and monitoring."""
    app.logger.info("Accessing index route '/'")
    # Use management API endpoints
    status_data = call_backend("GET", "/service/status")
    monitoring_data = call_backend("GET", "/monitoring/stats")
    popular_models_data = call_backend("GET", "/models/popular") # Still uses management prefix

    service_status = "Error Fetching"; service_enabled = "Error Fetching"
    models = []; stats = None; popular_models = []
    total_vram_gb = 0.0

    if isinstance(status_data, dict):
        service_status = status_data.get("service_status", "Unknown")
        service_enabled = status_data.get("service_enabled", "Unknown")
        models = status_data.get("configured_models", []) # This list now has ConfiguredModelInfo structure
    else: app.logger.warning("Failed to fetch service status from backend.")

    if isinstance(monitoring_data, dict):
        stats = monitoring_data
        gpu_stats = stats.get("gpu_stats")
        if isinstance(gpu_stats, list) and gpu_stats:
            total_vram_mb = sum(gpu.get("memory_total_mb", 0) for gpu in gpu_stats)
            total_vram_gb = total_vram_mb / 1024.0
            app.logger.info(f"Calculated total VRAM: {total_vram_gb:.2f} GB from {len(gpu_stats)} GPUs.")
        else: app.logger.warning("No valid GPU stats list found in monitoring data.")
    else: app.logger.warning("Failed to fetch monitoring stats from backend.")

    popular_params = {}
    if total_vram_gb > 0: popular_params["available_vram_gb"] = round(total_vram_gb, 2)
    popular_models_response = call_backend("GET", "/models/popular", params=popular_params)

    if isinstance(popular_models_response, list):
        popular_models = popular_models_response
        app.logger.info(f"Successfully fetched {len(popular_models)} popular models.")
    else: app.logger.warning("Failed to fetch popular models from backend or received unexpected format.")

    configured_model_ids = {m.get('model_id') for m in models}

    return render_template('index.html',
                           service_status=service_status,
                           service_enabled=service_enabled,
                           # active_model_key removed
                           models=models, # List of ConfiguredModelInfo
                           popular_models=popular_models,
                           total_vram_gb=total_vram_gb,
                           configured_model_ids=configured_model_ids,
                           monitoring_stats=stats,
                           config_path="model_config.json" # Still relevant for user info
                           )

# Systemd service action route removed. Ray Serve is managed automatically.

# Removed handle_activate_model route

@app.route('/download/<model_name>', methods=['GET', 'POST'])
def handle_download_model(model_name: str):
    """Handles download request for a specific *configured* model."""
    app.logger.info(f"Handling download request for configured model: {model_name}")
    if not model_name:
         flash("No model name provided for download.", "error")
         return redirect(url_for('index'))
    
    # Handle both GET and POST requests
    if request.method == 'POST':
        hf_token = request.form.get('hf_token')
        force_download = request.form.get('force') == 'on'
        payload = {"models": [model_name], "token": hf_token if hf_token else None, "force": force_download}
        result = call_backend("POST", "/models/download", json_data=payload)
        if result and isinstance(result, dict) and result.get("status") == "ok":
            flash(f"Download task for model '{model_name}' started in background.", "success")
        return redirect(url_for('index'))
    else:
        # For GET requests, show a download form
        return render_template('download.html', model_name=model_name)

# Renamed route
@app.route('/add_model/<path:model_id>', methods=['POST'])
def handle_add_model(model_id: str):
    """Handles request to add a popular model (by ID) to the config."""
    app.logger.info(f"Handling request to add popular model ID: {model_id}")
    if not model_id:
         flash("No model ID provided for adding.", "error")
         return redirect(url_for('index'))
    payload = {"model_ids": [model_id]}
    result = call_backend("POST", "/config/models", json_data=payload)
    if result and isinstance(result, dict) and result.get("status") == "ok":
        message = result.get('message', f"Request to add model ID '{model_id}' processed.")
        flash(message, "success")
        # Optional: Trigger download automatically? No, keep separate for now.
        # added_keys = result.get("details", {}).get("added_keys", [])
        # if added_keys: handle_download_model(added_keys[0]) # Needs adjustment
    elif result and isinstance(result, dict) and result.get("status") == "skipped":
         flash(result.get('message', f"Model '{model_id}' was already configured."), "warning")
    # Error flashing handled by call_backend
    return redirect(url_for('index'))

@app.route('/toggle_serve/<model_key>', methods=['POST'])
def handle_toggle_serve(model_key: str):
    """Handles toggling the serve status for a configured model."""
    app.logger.info(f"Handling request to toggle serve status for model key: {model_key}")
    if not model_key:
         flash("No model key provided for toggling serve status.", "error")
         return redirect(url_for('index'))
    # Determine the desired state (toggle: if current is true, send false, vice versa)
    # Requires fetching current state first - simpler to pass desired state from form
    serve_action = request.form.get('serve_action', 'false') # Default to disabling if form value missing
    serve_bool = serve_action.lower() == 'true'

    payload = {"serve": serve_bool}
    result = call_backend("PUT", f"/config/models/{model_key}/serve", json_data=payload)

    if result and isinstance(result, dict) and result.get("status") == "ok":
        flash(result.get('message', f"Serve status for '{model_key}' updated. Ray Serve redeployed."), "success")
    # Error flashing handled by call_backend
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
    app.run(host='0.0.0.0', port=5000, debug=True)