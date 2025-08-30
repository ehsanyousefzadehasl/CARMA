import os
import subprocess
import psutil

def get_gpu_process_info():
    try:
        # Get GPU processes using nvidia-smi
        result = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        gpu_processes = [line.split(",") for line in result.splitlines()]
        return gpu_processes
    except Exception as e:
        print(f"Error while running nvidia-smi: {e}")
        return []

def find_python_scripts():
    gpu_processes = get_gpu_process_info()

    gpus = []
    # Iterate over each process found by nvidia-smi
    for gpu_uuid, pid, process_name in gpu_processes:
        pid = pid.strip()
        process_name = process_name.strip()

        # Only consider Python processes
        if 'python' in process_name.lower():
            try:
                # Get the command line of the process using psutil
                process = psutil.Process(int(pid))
                cmdline = process.cmdline()

                # Check if the script name contains "xlnet" or "gpt"
                if cmdline and any(keyword in cmdline[1].lower() for keyword in ['xlnet', 'gpt']):
                    script_name = cmdline[1] if len(cmdline) > 1 else '<unknown>'
                    gpus.append(gpu_uuid)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError) as e:
                print(f"Could not access process details for PID {pid}: {e}")
    return gpus
