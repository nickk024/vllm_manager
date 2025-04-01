#!/usr/bin/env python3
"""
GPU monitoring script for vLLM
"""
import os
import json
import time
import logging
import argparse
import subprocess
import requests
from datetime import datetime
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_monitor")

def parse_args():
    parser = argparse.ArgumentParser(description="GPU Monitoring Tool for vLLM")
    parser.add_argument("--output", type=str, required=True, help="Output file for metrics")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--dashboard", action="store_true", help="Enable live dashboard output")
    return parser.parse_args()

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi or NVML"""
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            gpu_stats = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_stats.append({
                    "gpu_id": i,
                    "name": name,
                    "memory_used_mb": mem_info.used / (1024 * 1024),
                    "memory_total_mb": mem_info.total / (1024 * 1024),
                    "memory_utilization_pct": (mem_info.used / mem_info.total) * 100,
                    "gpu_utilization_pct": util.gpu,
                    "temperature_c": temp
                })
            
            pynvml.nvmlShutdown()
            return gpu_stats
        except Exception as e:
            logger.error(f"Error getting GPU stats via NVML: {e}")
    
    # Fallback to nvidia-smi
    try:
        output = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], text=True)
        
        gpu_stats = []
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpu_stats.append({
                    "gpu_id": int(parts[0]),
                    "name": parts[1],
                    "temperature_c": float(parts[2]),
                    "gpu_utilization_pct": float(parts[3]),
                    "memory_used_mb": float(parts[4]),
                    "memory_total_mb": float(parts[5]),
                    "memory_utilization_pct": (float(parts[4]) / float(parts[5])) * 100 if float(parts[5]) > 0 else 0
                })
        
        return gpu_stats
    except Exception as e:
        logger.error(f"Error getting GPU stats via nvidia-smi: {e}")
        return []

def get_vllm_stats(api_url):
    """Get statistics from vLLM API"""
    try:
        response = requests.get(f"{api_url}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get models: HTTP {response.status_code}")
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.error(f"Error connecting to vLLM API: {e}")
        return {"error": str(e)}

def get_system_stats():
    """Get system statistics (CPU, memory)"""
    try:
        # Get CPU usage
        cpu_percent = subprocess.check_output("top -bn1 | grep 'Cpu(s)' | awk '{print $2+$4}'", shell=True, text=True).strip()
        
        # Get memory usage
        mem_info = subprocess.check_output("free -m | grep Mem:", shell=True, text=True).strip().split()
        mem_total = float(mem_info[1])
        mem_used = float(mem_info[2])
        mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
        
        return {
            "cpu_utilization_pct": float(cpu_percent),
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
            "memory_utilization_pct": mem_percent
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {}

def display_metrics(metrics):
    """Display current metrics in a readable format."""
    print(f"\n=== vLLM Monitoring: {metrics['timestamp']} ===")
    
    # Display GPU stats
    print("\nGPU STATS:")
    for gpu in metrics.get("gpu_stats", []):
        print(f"  GPU {gpu['gpu_id']} ({gpu['name']}):")
        print(f"    Memory: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB ({gpu['memory_utilization_pct']:.1f}%)")
        print(f"    Utilization: {gpu['gpu_utilization_pct']:.1f}%")
        print(f"    Temperature: {gpu['temperature_c']}°C")
    
    # Display system stats
    system = metrics.get("system_stats", {})
    if system:
        print("\nSYSTEM STATS:")
        print(f"  CPU Utilization: {system.get('cpu_utilization_pct', 'N/A'):.1f}%")
        print(f"  Memory: {system.get('memory_used_mb', 'N/A'):.0f}MB / {system.get('memory_total_mb', 'N/A'):.0f}MB ({system.get('memory_utilization_pct', 'N/A'):.1f}%)")
    
    # Display vLLM API stats
    api_stats = metrics.get("vllm_stats", {})
    if "error" not in api_stats:
        print("\nvLLM API STATS:")
        if "data" in api_stats and api_stats["data"]:
            for model in api_stats["data"]:
                print(f"  Model: {model.get('id', 'Unknown')}")
        else:
            print("  No models currently loaded")
    else:
        print(f"\nvLLM API: Not available ({api_stats['error']})")

def collect_metrics(api_url):
    """Collect and combine all metrics."""
    timestamp = datetime.now().isoformat()
    
    metrics = {
        "timestamp": timestamp,
        "gpu_stats": get_gpu_stats(),
        "system_stats": get_system_stats(),
        "vllm_stats": get_vllm_stats(api_url)
    }
    
    return metrics

def main():
    args = parse_args()
    output_file = args.output
    interval = args.interval
    api_url = args.api_url
    show_dashboard = args.dashboard
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    logger.info(f"Starting GPU monitoring every {interval} seconds")
    logger.info(f"Metrics will be saved to {output_file}")
    logger.info(f"Monitoring vLLM API at {api_url}")
    
    try:
        while True:
            metrics = collect_metrics(api_url)
            
            # Save metrics to file
            with open(output_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            
            # Display dashboard if enabled
            if show_dashboard:
                os.system("clear")  # Clear terminal
                display_metrics(metrics)
            
            # Sleep until next collection
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
