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
LOG_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_DIR = os.environ.get("VLLM_LOG_DIR", LOG_DIR_DEFAULT)
LOG_FILE = os.path.join(LOG_DIR, "vllm_frontend.log")
LOG_LEVEL = logging.DEBUG
log_dir_valid = False
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_dir_valid = True
    print(f"[Flask INFO] Logging directory set to: {LOG_DIR}")
except OSError as e:
     print(f"[Flask ERROR] Could not create/access log directory: {LOG_DIR}. Error: {e}", file=sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app.logger.setLevel(LOG_LEVEL)
for handler in app.logger.handlers[:]: app.logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
app.logger.addHandler(console_handler)
if log_dir_valid:
    try:
        file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
        app.logger.info(f"--- Flask Logging configured (Console: INFO, File: {LOG_LEVEL} at {LOG_FILE}) ---")
    except Exception as e:
         print(f"[Flask ERROR] Failed to create file log handler for {LOG_FILE}. Error: {e}", file=sys.stderr)
         app.logger.error(f"Failed to create file log handler for {LOG_FILE}. Error: {e}")
else:
     app.logger.warning("--- Flask Logging configured (Console only) ---")
# --- End Logging Configuration ---

BACKEND_API_URL = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8080/api/v1")

def call_backend(method: str, endpoint: str, params: dict = None, json_data=None) -> dict | None:
    """Helper function to call the backend API. Added params for GET requests."""
    url = f"{BACKEND_API_URL}{endpoint}"
    action_description = f"{method} {endpoint}"
    try:
        app.logger.info(f"Calling backend: {action_description} with params: {params}, data: {json_data}")
        response = requests.request(method, url, params=params, json=json_data, timeout=30) # Pass params for GET
        response.raise_for_status()
        if response.status_code == 204:
             app.logger.info(f"Backend call successful ({action_description}): Status 204 No Content")
             return {"status": "ok", "message": "Action successful (no content)."}
        try:
            response_json = response.json()
            app.logger.info(f"Backend call successful ({action_description}): {response_json}")
            # Check if it's a list (like from /models/popular) or a dict
            if isinstance(response_json, list):
                 # If it's a list, wrap it for consistency downstream if needed, or handle directly
                 return {"status": "ok", "message": "List received.", "data": response_json}
            elif 'status' not in response_json or 'message' not in response_json:
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
    status_data = call_backend("GET", "/service/status")
    monitoring_data = call_backend("GET", "/monitoring/stats")

    # Initialize data
    service_status = "Error Fetching"; service_enabled = "Error Fetching"
    active_model_key = "Error Fetching"; models = []; stats = None
    popular_models = []
    total_vram_gb = 0.0

    if status_data:
        service_status = status_data.get("service_status", "Unknown")
        service_enabled = status_data.get("service_enabled", "Unknown")
        active_model_key = status_data.get("active_model_key", "N/A")
        models = status_data.get("configured_models", [])
    else: app.logger.warning("Failed to fetch service status from backend.")

    if monitoring_data:
        stats = monitoring_data
        gpu_stats = stats.get("gpu_stats", [])
        if gpu_stats:
            total_vram_mb = sum(gpu.get("memory_total_mb", 0) for gpu in gpu_stats)
            total_vram_gb = total_vram_mb / 1024.0
            app.logger.info(f"Calculated total VRAM: {total_vram_gb:.2f} GB")
        else: app.logger.warning("No GPU stats found in monitoring data to calculate total VRAM.")
    else: app.logger.warning("Failed to fetch monitoring stats from backend.")

    # Fetch popular models, passing available VRAM
    popular_params = {}
    if total_vram_gb > 0:
         popular_params["available_vram_gb"] = round(total_vram_gb, 2)
    # Can also add top_n parameter if needed: popular_params["top_n"] = 10
    popular_models_response = call_backend("GET", "/models/popular", params=popular_params)

    if popular_models_response and popular_models_response.get("status") == "ok":
        # Assuming backend returns list in 'data' field if wrapped, or directly if not
        popular_models = popular_models_response.get('data', popular_models_response if isinstance(popular_models_response, list) else [])
    else: app.logger.warning("Failed to fetch popular models from backend.")

    configured_model_ids = {m.get('model_id') for m in models}

    return render_template('index.html',
                           service_status=service_status,
                           service_enabled=service_enabled,
                           active_model_key=active_model_key,
                           models=models,
                           popular_models=popular_models,
                           total_vram_gb=total_vram_gb,
                           configured_model_ids=configured_model_ids,
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
    """Handles download request for a specific configured model."""
    app.logger.info(f"Handling download request for model: {model_name}")
    if not model_name:
         flash("No model name provided for download.", "error")
         return redirect(url_for('index'))
    hf_token = request.form.get('hf_token')
    force_download = request.form.get('force') == 'on'
    payload = {"models": [model_name], "token": hf_token if hf_token else None, "force": force_download}
    result = call_backend("POST", "/models/download", json_data=payload)
    if result and result.get("status") == "ok":
        flash(f"Download task for model '{model_name}' started in background.", "success")
    return redirect(url_for('index'))

@app.route('/add_popular/<path:model_id>', methods=['POST'])
def handle_add_popular_model(model_id: str):
    """Handles request to add a popular model (by ID) to the config."""
    app.logger.info(f"Handling request to add popular model ID: {model_id}")
    if not model_id:
         flash("No model ID provided for adding.", "error")
         return redirect(url_for('index'))

    # The model_id might contain slashes, Flask's path converter handles this.
    # We pass it directly to the backend.
    payload = {"model_ids": [model_id]}
    result = call_backend("POST", "/config/models", json_data=payload)

    if result and result.get("status") == "ok":
        # Check how many were actually added (might have been skipped)
        message = result.get('message', f"Request to add model ID '{model_id}' processed.")
        flash(message, "success")
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