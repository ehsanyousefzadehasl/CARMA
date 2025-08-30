import time
import datetime
import csv
import pandas as pd
from subprocess import Popen, PIPE
from sklearn.cluster import KMeans
import numpy as np

import subprocess
import io

Gmetrics = pd.DataFrame(columns=['active', 'GPU_mem_in_use', 'GPU_mem_total', 'GPU_mem_available', 'smact'])

# executes a shell bash command and return the output
def execute_command(cmd, shell=False, vars={}):
    """
    output: string format terminal based output of a bash shell command
    
    Executes a bash command and return the result
    """
    cmd = cmd.split()
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True, shell=shell) as p:
        o=p.stdout.read()
    return o

# returns detected GPUs on the system excluding the one 
# that is specifically for dispaly purposes
def gpu_uuids():
    """
    output: dictionary of gpu_index:gpu_uuid
    
    Detects GPUs on the system using 'nvidia-smi'
    """
    gpus = dict()
    o = execute_command("nvidia-smi -L")
    o = o.split('\n')
    for line in o:
        # ignoring the display dedicated GPU on the DGX A100 station machine
        if "Display" in line:
            continue
        if "UUID: GPU" in line:
            gpu_uuid = line.split("UUID:")[1].split(")")[0].strip()
            gpus[gpu_uuid] = line.split("GPU")[1].split(":")[0].strip()

    return gpus


# print(gpu_uuids())

# exit()
# GPU_to_PID is a map showing which process are running on which GPUs
GPU_to_PID = dict()


# uses nvidi-smi tool to get memory usage of each GPU
def gpu_mem_usage():
    """
    output: dictionary of gpu_uuids:gpu_memory_usage

    Uses nvidia-smi to monitor available GPUs' memory usage
    """
    result = dict()

    o = execute_command("nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv")
    o = o.split('\n')
    
    for line in o:
        if line.startswith("uuid") or line == '':
            continue
        gpu_id, b, c = line.split(',')
        memory_usage = b.split(' ')[1] 
        memory_cap = c.split(' ')[1] 
        
        result[gpu_id] = [memory_usage, memory_cap]

    return result



# dcgmi monitor
def dcgmi_monitor():
    """
    output: dictionary of gpu_uuids:smact
    Uses dcgmi to monitor available GPUs and GPU MIG instances
    """

    result = dict()
    # ====== compute resources ========
    # 203 gpu utilization [0]
    # 1001 gract [1], 1002 smact [2], 1003 smocc [3], 1006 fp64 [4], 1007 fp32 [5], 1008 fp16 [6], 1004 tensor core [7]
    
    # ======== memory =========
    # 204 memory copy [8], 1005 drama [9]
    
    # ======= data transfer ============
    # 1009 pcie_tx_bytes [10], 1010 pcie_rx_bytes [11]
    # 1011 nvlink_tx_bytes [12], 1012 nvlink_rx_bytes [13]
    
    # ====== power and energy =======
    # 155 power [14], 156 energy [15]
    
    result = dict()

    o = execute_command("dcgmi dmon -e 203,1001,1002,1003,1006,1007,1008,1004,204,1005,1009,1010,1011,1012,155,156 -c 1")
    o = o.split("\n")

    for line in o:
        if line.startswith("#") or line.startswith("ID") or line == "":
            continue
        gpu_id = line.split("GPU")[1].split()[0].strip()
        metrics = line.split("GPU")[1].split()[1:]
        assert(len(metrics) == 16)
        result[gpu_id] = metrics 

    return result

def top_extractor(process_names = ["python"]):
    # they will keep the accumulative %CPU and %MEM for all wanted processes
    CPU_util = 0
    Mem_util = 0
    all_memory = 0
    free_available_system_memory = 0
    memory_used_total = 0

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    top = subprocess.Popen("top -i -b -n 1".split(), stdout=subprocess.PIPE)
    # print(top)

    for line in io.TextIOWrapper(top.stdout, encoding="utf-8"):
        line = line.lstrip()

        if line.startswith("top") or line.startswith("Tasks") or line.startswith("%") or line.startswith("PID") or line.startswith(" "):
                pass
        else:
            word_vector = line.strip().split()
            if (line.startswith("KiB") or line.startswith("MiB")) and len(word_vector) != 0:
                    if word_vector[1] == "Mem":
                        
                        if word_vector[4] == "total,":
                            all_memory = float(word_vector[3]) / 1000

                        if word_vector[6] == "free,":
                            free_available_system_memory = float(word_vector[5]) / 1000

                        if word_vector[8] == "used,":
                            memory_used_total = float(word_vector[7]) / 1000

            elif len(word_vector) != 0:
                    if word_vector[11] in process_names:
                        CPU_util += float(word_vector[8])
                        Mem_util += float(word_vector[9])

    to_return = pd.DataFrame(columns=["total_memory", "free_memory", "used_memory", 
                                                     "%CPU", "%MEM", "time"])
    data_to_return = [all_memory, free_available_system_memory, memory_used_total, CPU_util, Mem_util, now]

    to_return.loc[len(to_return)] = data_to_return
    return to_return

header_flag = True
# developed and tested
def monitor_and_log_minute_scale(log=False, timestamp=3, window=1):
    """
    - monitors detected GPUs with <timestep> seconds intervals for <window> number of minutes

    - uses 'gpu_mem_usage()' and 'dcgmi_monitor()' functions
    """
    flag = True
    start = datetime.datetime.now()
    
    window_monitored_metrics = pd.DataFrame(columns=["gpu_id", "gpu_uuid", "gpu_memory_usage", "gpu_memory_cap", "gpu_utilization", "gract", "smact",
                                                     "smocc", "fp64", "fp32", "fp16", "tc",
                                                     "memory_copy", "dram_active", "pcie_tx_bytes",
                                                     "pcie_rx_bytes", "nvlink_tx_bytes", "nvlink_rx_bytes",
                                                     "power", "energy"])

    while(flag):
        # ### HERE starts monitoring logic HERE ###
        gpus = gpu_uuids()
        gm = gpu_mem_usage()
        dcgm = dcgmi_monitor()
        res = []

        temp = pd.DataFrame(columns=["gpu_id", "gpu_uuid", "gpu_memory_usage", "gpu_memory_cap", "gpu_utilization", "gract", "smact",
                                                    "smocc", "fp64", "fp32", "fp16", "tc",
                                                    "memory_copy", "dram_active", "pcie_tx_bytes",
                                                    "pcie_rx_bytes", "nvlink_tx_bytes", "nvlink_rx_bytes",
                                                    "power", "energy"])
        all_metrics = []
        for gpu_uuid in gpus:
            gpu_id = gpus[gpu_uuid]
            gpu_memory_info = gm[gpu_uuid]

            # print(gpu_memory_info)

            dcgm_metrics = dcgm[gpu_id]
            all_metrics = [gpu_id] + [gpu_uuid] + gpu_memory_info + dcgm_metrics

            window_monitored_metrics.loc[len(window_monitored_metrics)] = all_metrics
            temp.loc[len(temp)] = all_metrics
            
            
        # the logic to log not excessive rows
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        temp1 = temp.assign(time=[now]*len(temp))
        global header_flag

        # ++++ writing to the csv log file if logging is enabled ++++
        if header_flag == True:
            temp1.to_csv('dcgmi_metrics.csv', mode='a', index=False)
            header_flag = False
        else:
            temp1.to_csv('dcgmi_metrics.csv', mode='a', header = False, index=False)


            # print(window_monitored_metrics)





        # ### HERE ends monitoring logic HERE ###
        # ***** HERE starts time control logic *****
        now = datetime.datetime.now()
        if datetime.timedelta(minutes=window) < (now - start):
            flag = False
        # ***** HERE ends days control logic *****

        # ====== timestep for sampling logic ======
        time.sleep(timestamp)
    
    return window_monitored_metrics


def gpus_activeness():
    """
    Analyzing 'nvidia-smi pmon -c 1' command and figuring GPUs status
    
    This has the responsibility of updating GPU_to_PID global variable 
    """
    gpus = gpu_uuids()

    # reversing keys and values in gpu_uuids dictionary
    # to be able to find the corresponding uuid
    tmp_gpus = {value: key for key, value in gpus.items()}    
    activity_dict = dict.fromkeys(tmp_gpus, 0)

    # active GPUs
    active_GPUs = dict()
    out = execute_command("nvidia-smi pmon -c 1")
    out = out.split("\n")

    
    for line in out:
        if line.startswith("#"):
            pass
        elif len(line) != 0:
            tmp_list = line.strip().split()
            # print(tmp_list)
                                   # tmp_list[0] shows GPU index
            if tmp_list[1] != '-' and tmp_list[7] != 'nvidia-cuda-mps':
                print(tmp_list[7])
                # tmp_list[1] contains PID of the process
                if tmp_list[0] in active_GPUs:
                    active_GPUs[tmp_list[0]].append(tmp_list[1])
                    # print(active_GPUs)
                else:
                    # print("it has one")
                    active_GPUs[tmp_list[0]] = [tmp_list[1]]
                    # print(active_GPUs)


    GPU_to_PID = active_GPUs

    # print(GPU_to_PID)
    # now we know which GPUs are active with their index
    
    for active_gpu_index in active_GPUs:
        activity_dict[active_gpu_index] = 1

    out_dict = dict()

    for index in tmp_gpus:
        tmp = tmp_gpus[index]
        out_dict[tmp] = activity_dict[index]

    return out_dict



def analyze(data = None):
    gpu_ids = gpu_uuids()                       # getting list of gpus
    gpus_activity = gpus_activeness()           # getting activeness of gpus

    analyzed_data = dict()                      # analyzed data

    grouped = data.groupby(["gpu_uuid"])        # grouping the monitored data based on gpu_uuid
    
    for uuid in gpu_ids:
        df1 = grouped.get_group(uuid)

        smact = np.array(df1.loc[:,"smact"].values, dtype=float)
        gpu_mem = np.array(df1.loc[:,"gpu_memory_usage"].values, dtype=np.float64)
        gpu_mem_cap = np.array(df1.loc[:,"gpu_memory_cap"].values, dtype=np.float64)
        
        smact = np.mean(smact)
        
        # smact = smact.reshape(-1, 1)
        # with warnings.catch_warnings():         # for ignoring warnings when the number of cluster is more than the actual available 
        #     warnings.simplefilter("ignore")    
        #     kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(smact)
        # # print(max(gpu_mem), smact, kmeans.labels_, kmeans.cluster_centers_, np.mean(smact))

        # # finding the dominant cluster
        # counter = Counter(kmeans.labels_)
        # largest_cluster_idx = np.argmax(counter.values())
        # largest_cluster_center = kmeans.cluster_centers_[largest_cluster_idx]


        # analyzed_data[uuid] = [gpus_activity[uuid], last gpu_mem, gpu_mem_cap[0], largest_cluster_center[0]] 
        analyzed_data[uuid] = [gpus_activity[uuid], gpu_mem[-1], gpu_mem_cap[0], (gpu_mem_cap[0] - gpu_mem[-1]),  smact] 
    df = pd.DataFrame.from_dict(analyzed_data)
    df = df.T
    df.rename(columns={0: 'active', 1: 'GPU_mem_in_use', 2: 'GPU_mem_total', 3: 'GPU_mem_available', 4: 'smact'}, inplace=True)
    return df

def metrics():
    while True:
        global Gmetrics
        a = monitor_and_log_minute_scale(log=False, timestamp=1, window=1.5)

        # ==== we must have GPU memory updated more frequently ======
        # gpu_memory = gpu_mem_usage()
    
        m = analyze(a)
        Gmetrics = m


# print(top_extractor())