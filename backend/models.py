from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- API Request/Response Models ---

class ModelInfo(BaseModel):
    """Information about a configured model."""
    name: str = Field(..., description="The configuration key for the model.")
    model_id: str = Field(..., description="The Hugging Face model ID.")
    downloaded: bool = Field(..., description="Whether the model files are present locally.")
    serve: Optional[bool] = Field(None, description="Whether the model is configured to be served (from model_config.json).")

class PopularModelInfo(BaseModel):
    """Information about a popular/curated model."""
    model_id: str
    name: str
    size_gb: float
    gated: bool
    config: Dict[str, Any] # Base config details

class DownloadRequest(BaseModel):
    """Request body for downloading models."""
    models: Optional[List[str]] = Field(None, description="List of model names (from config) to download. If None, downloads all.")
    token: Optional[str] = Field(None, description="Hugging Face token for gated models.")
    force: Optional[bool] = Field(False, description="Force re-download even if model exists.")

class AddModelRequest(BaseModel):
    """Request body for adding models to the configuration."""
    model_ids: List[str] = Field(..., description="List of model_id strings (e.g., 'meta-llama/Llama-3-8B-Instruct') to add from the popular list.")

class ServiceActionResponse(BaseModel):
    """Response body for service actions."""
    status: str = Field(..., description="'ok' or 'error'")
    message: str
    details: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary for additional details.")

class GPUStat(BaseModel):
    """Statistics for a single GPU."""
    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization_pct: float
    gpu_utilization_pct: float
    temperature_c: float

class SystemStats(BaseModel):
    """System-level statistics (CPU/Memory)."""
    cpu_utilization_pct: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    memory_utilization_pct: Optional[float] = None

class MonitoringStats(BaseModel):
    """Combined monitoring statistics."""
    timestamp: str
    gpu_stats: List[GPUStat]
    system_stats: SystemStats

class ConfiguredModelStatus(BaseModel):
    """Detailed status including Ray Serve status and configured models."""
    ray_serve_status: str = Field(..., description="Status of Ray Serve (e.g., 'running', 'not_running', 'error').")
    configured_models: List[ModelInfo] = Field(..., description="List of all models in the configuration, including their download and intended serve status.")

class GeneralResponse(BaseModel):
    """Generic response model."""
    status: str = Field(..., description="General status indicator (e.g., 'ok', 'skipped', 'error').")
    message: str = Field(..., description="A descriptive message about the operation.")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary for additional details (e.g., added keys).") # Added details here too

class ApiTestResponse(BaseModel):
    """Response model for the API test endpoint."""
    status: str
    message: str
    vllm_api_reachable: bool
    vllm_response: Optional[Any] = None # Store the raw response from vLLM
    error_details: Optional[str] = None

class ToggleServeRequest(BaseModel):
    """Request body for toggling the serve status of a model."""
    serve: bool = Field(..., description="Set to true to serve the model, false to unserve.")