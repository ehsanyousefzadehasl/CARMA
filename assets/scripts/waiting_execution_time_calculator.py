import os
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import re

# waiting time section
job_queued_dispatched_dic = dict()

# f = "001-exclusive"
# f = "002-oracle-first-fit"
# f = "002-oracle-first-fit-balance"
# f = "001-exclusive-balance"
# f = "002-oracle-first-fit-balance-MPS"
# f = "003-oracle-first-fit-balance-MPS-multi-gpu-alone-with-fragmentation-OOm"
# f = "002-oracle-first-fit-balance-multi-gpu-alone"

# f = "01-oracle-first-fit"
# f = "00-exclusive"
# f = "02-most-gpumem-avail-1oom"
# f = "02-most-gpumem-avail-1oom-with-looser-recovery-policy"
# f = "00-exclusive-reordered"

# f = "01-MGMEM"
# f = "02-MGMEM-MPS"
# f = "03-FF-MPS"
# f = "04-BF-MPS"
# f = "05-LGU-MPS"
# f = "06-MGMEM-MPS-RR"

# f = "07-LGU-MPS-RR"

# f = "08-MGMEM-MPS_RR-smact75"

# f = "09-MGMEM-MPS_RR-smact85"

# f = "10-Horus-MGMEM-80%smact"

# f = "11-Faketensor-GMEM-80%smact"

# f = "12-estimator-MGMEM-80%smact"

# f = "13-horus-MGMEM"

# f = "14-faketensor-MGMEM"

# f = "15-GPUMemNet-MGMEM"

# f = "16-MGMEM"

# f = "17-MGMEM-smact80%"

# f = "18-MGMEM-smact80%-gpumem2g"


# f = "19-round-robin"

# f = "20-oracle-MGMEM-smact70%-2gig-MPS"

# f = "21-oracle-MGMEM-smact70-2gig-stream"

# f = "00-exclusive"

# f = "01-round-robin"

# f = "02-round-robin-MPS"

# f = "03-MGMEM-MPS"

# f = "04-LGUTIL-MPS"

# f = "05-horus"

# f = "06-faketensor"

f = "07-GPUMemNet"

file = open(f'/home/ehyo/rad-scheduler/{f}/std.log', 'r')
Lines = file.readlines()

for line in Lines:
    temp = line.split()
    ttemp = temp[0]+" "+temp[1]

    # print(temp[3])
    # changing it to python datetime object for the sake of easier calculation 
    date_format = '%d-%b-%y %H:%M:%S'
    time_point = datetime.strptime(ttemp, date_format)
    
    if temp[2] == "queued":
        job_queued_dispatched_dic[temp[3]] = [time_point]
    elif temp[2] == "dispatched":
        job_queued_dispatched_dic[temp[3]].append(time_point)

waiting_time = pd.DataFrame(columns=['task_id', 'waiting_time(s)', 'waiting_time(m)', 'waiting_time(h)'])
for key in job_queued_dispatched_dic:
    print(job_queued_dispatched_dic)
    waiting_time_seconds = (job_queued_dispatched_dic[key][len(job_queued_dispatched_dic[key])-1] - job_queued_dispatched_dic[key][0]).total_seconds()
    waiting_time_minutes = waiting_time_seconds/ 60
    waiting_time_hours   = waiting_time_minutes/ 60
    waiting_time.loc[len(waiting_time)] = [key, waiting_time_seconds, waiting_time_minutes, waiting_time_hours]

exe_time_dic = dict()

date_format = '%H:%M:%S'
epoch_time = datetime(1970, 1, 1)

list_of_files = []
for file in os.listdir(f"/home/ehyo/rad-scheduler/{f}"):
    if file.startswith("time") and file.endswith(".et"):
        file = os.path.join(f"/home/ehyo/rad-scheduler/{f}", file)
        list_of_files.append(file)

counter = 0
for iterator in list_of_files:
    file = open(f'{iterator}', 'r')
    Lines = file.readlines()

    pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'

    match = re.search(pattern, iterator)

    task_id = match.group(0)

    if len(Lines)!=0:
        for line in Lines:
            if "real" in line:
                a = line.strip().split()[1]
                minutes, rest = a.split("m")
                seconds = rest.split("s")[0]
                total_time_in_seconds = (float(minutes) * 60 + float(seconds))
                total_time_in_minutes = total_time_in_seconds/ 60
                total_time_in_hours = total_time_in_minutes/ 60

                counter += 1
                if task_id in exe_time_dic.keys():
                    exe_time_dic[task_id][0] += total_time_in_seconds
                    exe_time_dic[task_id][1] += total_time_in_minutes
                    exe_time_dic[task_id][2] += total_time_in_hours
                else:
                    exe_time_dic[task_id] = [total_time_in_seconds, total_time_in_minutes, total_time_in_hours]
            else:
                continue
    else:
        exe_time_dic[task_id] = [0, 0, 0]


print(len(exe_time_dic))

exe_time = pd.DataFrame(columns=['task_id', 'execution_time(s)', 'execution_time(m)', 'execution_time(h)'])
for key in exe_time_dic:
    waiting_time_seconds = exe_time_dic[key][0]
    waiting_time_minutes = exe_time_dic[key][1]
    waiting_time_hours   = exe_time_dic[key][2]
    exe_time.loc[len(exe_time)] = [key, waiting_time_seconds, waiting_time_minutes, waiting_time_hours]

merged_df = waiting_time.merge(exe_time, on='task_id', how='left')

merged_df["job_completion_time(s)"] = merged_df["execution_time(s)"] + merged_df["waiting_time(s)"]
merged_df["job_completion_time(m)"] = merged_df["execution_time(m)"] + merged_df["waiting_time(m)"]
merged_df["job_completion_time(h)"] = merged_df["execution_time(h)"] + merged_df["waiting_time(h)"]

print(merged_df)
df = merged_df.mean()
print(df.to_csv("summary.csv"))


# plotting part
        

# summarizing the info
# 1. 


plt.figure(figsize=(12, 6))
plt.bar(merged_df['task_id'], merged_df['waiting_time(m)'], label="waiting time (minutes)", color="blue", alpha=0.6)
plt.bar(merged_df['task_id'], merged_df['execution_time(m)'], label="exeution time (minutes)", color="green", alpha=0.6)
plt.scatter(merged_df['task_id'], merged_df['waiting_time(m)'] + merged_df['execution_time(m)'], label="Job Completino time (minutes)", color="red", lw=2)

plt.xticks(rotation=90, ha='right')
plt.legend()
plt.ylim(0, 1000)
plt.xlabel("Job IDs submiited over time")
plt.ylabel("Time (minutes)")
plt.savefig("wait_exe_time.png")