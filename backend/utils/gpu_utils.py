import subprocess
import logging
from typing import List
from ..models import GPUStat # Relative import for Pydantic model

logger = logging.getLogger(__name__)

# Attempt to import and initialize pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    logger.info("PyNVML initialized successfully.")
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("PyNVML not found. GPU stats will rely on nvidia-smi fallback.")
except pynvml.NVMLError as e:
    NVML_AVAILABLE = False
    logger.error(f"Failed to initialize PyNVML: {e}. GPU stats will rely on nvidia-smi fallback.")

def _get_gpu_stats_nvml() -> List[GPUStat]:
    """Get GPU stats using PyNVML."""
    gpu_stats = []
    if not NVML_AVAILABLE: # Should not happen if called correctly, but safety check
        return []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Fix: Assume nvmlDeviceGetName returns str directly (common in newer versions)
            # Remove .decode('utf-8')
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes): # Add check in case older versions return bytes
                 name = name.decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            gpu_stats.append(GPUStat(
                gpu_id=i, name=name,
                memory_used_mb=mem_info.used / (1024 * 1024),
                memory_total_mb=mem_info.total / (1024 * 1024),
                memory_utilization_pct=(mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0,
                gpu_utilization_pct=float(util.gpu),
                temperature_c=float(temp)
            ))
    except pynvml.NVMLError as e:
        logger.error(f"Error getting GPU stats via NVML: {e}. Trying nvidia-smi.")
        # Explicitly call the smi function as fallback within this util module
        return _get_gpu_stats_smi()
    except Exception as e:
         logger.error(f"Unexpected error in _get_gpu_stats_nvml: {e}", exc_info=True) # Log traceback
         return [] # Return empty on unexpected errors
    return gpu_stats

def _get_gpu_stats_smi() -> List[GPUStat]:
    """Get GPU stats using nvidia-smi (fallback)."""
    gpu_stats = []
    try:
        command = [
            'nvidia-smi',
            '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ]
        logger.debug(f"Running command: {' '.join(command)}")
        output = subprocess.check_output(command, text=True, timeout=5) # Add timeout
        for line in output.strip().split('\n'):
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                try:
                    mem_used = float(parts[4])
                    mem_total = float(parts[5])
                    gpu_stats.append(GPUStat(
                        gpu_id=int(parts[0]), name=parts[1],
                        temperature_c=float(parts[2]), gpu_utilization_pct=float(parts[3]),
                        memory_used_mb=mem_used, memory_total_mb=mem_total,
                        memory_utilization_pct=(mem_used / mem_total) * 100 if mem_total > 0 else 0
                    ))
                except ValueError as ve:
                    logger.warning(f"Could not parse nvidia-smi line: '{line}'. Error: {ve}")
            else:
                 logger.warning(f"Unexpected number of parts in nvidia-smi line: '{line}'")
    except FileNotFoundError:
        logger.error("nvidia-smi command not found. Cannot get GPU stats.")
    except subprocess.TimeoutExpired:
         logger.error("nvidia-smi command timed out.")
    except subprocess.CalledProcessError as e:
         logger.error(f"nvidia-smi command failed with exit code {e.returncode}: {e.stderr}")
    except Exception as e:
        logger.error(f"Error getting GPU stats via nvidia-smi: {e}", exc_info=True) # Log traceback
    return gpu_stats

def get_gpu_stats() -> List[GPUStat]:
    """Public function to get GPU stats, preferring NVML."""
    if NVML_AVAILABLE:
        logger.debug("Attempting GPU stats via NVML.")
        return _get_gpu_stats_nvml()
    else:
        logger.info("NVML not available, using nvidia-smi for GPU stats.")
        return _get_gpu_stats_smi()

def get_gpu_count() -> int:
    """Get number of available NVIDIA GPUs."""
    if NVML_AVAILABLE:
        try:
            count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Detected {count} GPUs via PyNVML.")
            return count
        except pynvml.NVMLError as e:
            logger.error(f"NVML error getting device count: {e}. Falling back to nvidia-smi.")

    # Fallback or if NVML not available/failed
    try:
        command = ["nvidia-smi", "--list-gpus"]
        logger.debug(f"Running command: {' '.join(command)}")
        output = subprocess.check_output(command, text=True, timeout=5) # Add timeout
        # Count non-empty lines
        count = len([line for line in output.strip().split("\n") if line.strip()])
        logger.info(f"Detected {count} GPUs via nvidia-smi.")
        return count
    except FileNotFoundError:
        logger.warning("nvidia-smi not found. Assuming 0 GPUs for count.")
        return 0
    except subprocess.TimeoutExpired:
         logger.error("nvidia-smi --list-gpus command timed out. Assuming 0 GPUs.")
         return 0
    except subprocess.CalledProcessError as e:
         logger.error(f"nvidia-smi --list-gpus failed with exit code {e.returncode}: {e.stderr}. Assuming 0 GPUs.")
         return 0
    except Exception as e:
        logger.error(f"Error running nvidia-smi for count: {e}. Assuming 0 GPUs.")
        return 0

# Optional: Add NVML shutdown hook if needed, though FastAPI handles shutdown
# import atexit
# if NVML_AVAILABLE:
#     atexit.register(pynvml.nvmlShutdown)
#     logger.info("Registered PyNVML shutdown hook.")