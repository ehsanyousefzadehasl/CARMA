import os
from datetime import datetime
import re
import pandas as pd

# Define the directory containing the .et files
# directory = "/home/ehyo/rad-scheduler/001-exclusive"
# directory = "/home/ehyo/rad-scheduler/002-oracle-first-fit"
# directory = "/home/ehyo/rad-scheduler/002-oracle-first-fit-balance"
# directory = "/home/ehyo/rad-scheduler/001-exclusive-balance"
# directory = "/home/ehyo/rad-scheduler/002-oracle-first-fit-balance-MPS"
# directory = "/home/ehyo/rad-scheduler/003-oracle-first-fit-balance-MPS-multi-gpu-alone"
# directory = "../003-oracle-first-fit-balance-MPS-multi-gpu-alone-with-fragmentation-OOm"
# directory = "../002-oracle-first-fit-balance-multi-gpu-alone"

# directory = "../01-oracle-first-fit"
# directory = "../00-exclusive"

# directory = "../02-most-gpumem-avail-1oom"

# directory = "../02-most-gpumem-avail-1oom-with-looser-recovery-policy"

# directory = "../00-exclusive-reordered"

# directory = "../01-MGMEM"

# directory = "../02-MGMEM-MPS"

# directory = "../03-FF-MPS"

# directory = "../04-BF-MPS"

# directory = "../05-LGU-MPS"

# directory = "../06-MGMEM-MPS-RR"

# directory = "../07-LGU-MPS-RR"

# directory = "../08-MGMEM-MPS_RR-smact75"

# directory = "../09-MGMEM-MPS_RR-smact85"

# directory = "../10-Horus-MGMEM-80%smact"

# directory = "../11-Faketensor-GMEM-80%smact"

# directory = "../12-estimator-MGMEM-80%smact"

# directory = "../13-horus-MGMEM"

# directory = "../14-faketensor-MGMEM"

# directory = "../15-GPUMemNet-MGMEM"

# directory = "../16-MGMEM"

# directory = "../17-MGMEM-smact80%"

# directory = "../18-MGMEM-smact80%-gpumem2g"

# directory = "../19-round-robin"

# directory = "../20-oracle-MGMEM-smact70%-2gig-MPS"

# directory = "../21-oracle-MGMEM-smact70-2gig-stream"

# directory = "../00-exclusive"

# directory = "../01-round-robin"

# directory = "../02-round-robin-MPS"

# directory = "../03-MGMEM-MPS"

# directory = "../04-LGUTIL-MPS"

# directory = "../05-horus"

# directory = "../06-faketensor"

# directory = "../07-GPUMemNet"

directory = "../08-oracle-MGMEM"

# Step 1: Find all .et files and sort them based on the timestamp in the filename
list_of_files = []
for file in os.listdir(directory):
    if file.startswith("time") and file.endswith(".et"):
        file_path = os.path.join(directory, file)
        list_of_files.append(file_path)

# Sort the list of files based on the timestamp in the filename
list_of_files.sort(key=lambda x: datetime.strptime(re.search(r'time-(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', x).group(1), '%Y-%m-%d_%H:%M:%S'))

# Step 2: Extract the first file and the last 20 files
first_file = list_of_files[0]
last_files = list_of_files[-20:]

# Step 3: Calculate the total duration
# Get the start time from the first file's name
start_time_str = re.search(r'time-(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', first_file).group(1)
start_time = datetime.strptime(start_time_str, '%Y-%m-%d_%H:%M:%S')

# Determine the latest end time from the last 20 files
max_end_time = None
for last_file in last_files:
    # Get the end time from the file's name
    end_time_str = re.search(r'time-(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', last_file).group(1)
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d_%H:%M:%S')

    # Open the file and read the execution time
    with open(last_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "real" in line:
                a = line.strip().split()[1]
                minutes, rest = a.split("m")
                seconds = rest.split("s")[0]
                execution_seconds = float(minutes) * 60 + float(seconds)
                end_time += pd.to_timedelta(execution_seconds, unit='s')

    # Update the maximum end time
    if max_end_time is None or end_time > max_end_time:
        max_end_time = end_time

# Step 4: Calculate the total duration from the start of the first task to the end of the latest finishing task
total_duration = (max_end_time - start_time).total_seconds()
total_duration_minutes = total_duration / 60
total_duration_hours = total_duration_minutes / 60

# Print the total duration
print(f"Total Duration: {total_duration:.2f} seconds ({total_duration_minutes:.2f} minutes, {total_duration_hours:.2f} hours)")
