import logging
from datetime import datetime
from fastapi import APIRouter

# Import models and utils using relative paths
from ..models import MonitoringStats
from ..utils.gpu_utils import get_gpu_stats
from ..utils.system_utils import get_system_stats

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/monitoring", # Prefix all routes in this router
    tags=["Monitoring"] # Tag for OpenAPI docs
)

@router.get("/stats", response_model=MonitoringStats, summary="Get System and GPU Stats")
async def get_monitoring_stats():
    """
    Retrieves current GPU statistics (using PyNVML or nvidia-smi)
    and system statistics (CPU/Memory using top/free).
    """
    timestamp = datetime.now().isoformat()
    logger.info("Collecting monitoring stats...")

    # Collect stats using utility functions
    gpu = get_gpu_stats()
    system = get_system_stats()

    logger.info(f"Collected {len(gpu)} GPU stats and system stats.")

    return MonitoringStats(
        timestamp=timestamp,
        gpu_stats=gpu,
        system_stats=system
    )