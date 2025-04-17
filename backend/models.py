from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- API Request/Response Models ---

class ModelInfo(BaseModel):
    """Information about a configured model."""
    name: str
    model_id: str
    downloaded: bool

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
    details: Optional[str] = None

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
    """Detailed status including service, active model, and configured models."""
    service_status: str = Field(..., description="Current status from systemctl (e.g., 'active', 'inactive', 'failed')")
    service_enabled: str = Field(..., description="Enabled status from systemctl (e.g., 'enabled', 'disabled')")
    active_model_key: Optional[str] = Field(None, description="The key of the model currently set as active in active_model.txt, or null if none.")
    configured_models: List[ModelInfo]

class GeneralResponse(BaseModel):
    """Generic response model."""
    status: str
    message: str

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