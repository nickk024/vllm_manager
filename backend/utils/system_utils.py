import subprocess
import logging
from ..models import SystemStats # Relative import for Pydantic model

logger = logging.getLogger(__name__)

def get_system_stats() -> SystemStats:
    """Get system statistics (CPU, memory) using shell commands."""
    stats = {}
    # --- Get CPU Usage ---
    try:
        # Using 'top' can be resource-intensive and might vary slightly across systems.
        # Consider 'psutil' library for a more robust and cross-platform approach if dependencies are acceptable.
        # This command gets the sum of user (%us) and system (%sy) CPU time from the 'top' output.
        cpu_command = "top -bn1 | grep '%Cpu(s)' | awk '{print $2+$4}'"
        logger.debug(f"Running command: {cpu_command}")
        cpu_output = subprocess.check_output(cpu_command, shell=True, text=True).strip()
        stats["cpu_utilization_pct"] = float(cpu_output)
    except FileNotFoundError:
        logger.warning("'top' command not found. Cannot get CPU stats.")
    except subprocess.CalledProcessError as e:
         logger.warning(f"Command '{cpu_command}' failed: {e}. Cannot get CPU stats.")
    except ValueError as e:
         logger.warning(f"Could not parse CPU stats output: {e}")
    except Exception as e:
        logger.warning(f"Could not get CPU stats using 'top': {e}")

    # --- Get Memory Usage ---
    try:
        # Using 'free -m' provides memory in Megabytes.
        mem_command = "free -m"
        logger.debug(f"Running command: {mem_command}")
        mem_output_lines = subprocess.check_output(mem_command, shell=True, text=True).strip().split('\n')
        # Find the 'Mem:' line (usually the second line)
        mem_line = None
        for line in mem_output_lines:
            if line.startswith("Mem:"):
                mem_line = line
                break

        if mem_line:
            mem_info = mem_line.split()
            # Expected format: Mem: total used free shared buff/cache available
            if len(mem_info) >= 3:
                mem_total = float(mem_info[1])
                mem_used = float(mem_info[2]) # 'used' column often includes buffers/cache depending on 'free' version
                stats["memory_total_mb"] = mem_total
                stats["memory_used_mb"] = mem_used
                stats["memory_utilization_pct"] = (mem_used / mem_total) * 100 if mem_total > 0 else 0
            else:
                logger.warning(f"Unexpected format in 'free -m' output line: {mem_line}")
        else:
             logger.warning("Could not find 'Mem:' line in 'free -m' output.")

    except FileNotFoundError:
        logger.warning("'free' command not found. Cannot get memory stats.")
    except subprocess.CalledProcessError as e:
         logger.warning(f"Command '{mem_command}' failed: {e}. Cannot get memory stats.")
    except ValueError as e:
         logger.warning(f"Could not parse memory stats output: {e}")
    except Exception as e:
        logger.warning(f"Could not get memory stats using 'free': {e}")

    return SystemStats(**stats)