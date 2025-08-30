import os
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Waiting time section
job_queued_dispatched_dic = dict()

# Choose the log folder
# f = "01-MGMEM"
# f = "02-MGMEM-MPS"
# f = "03-FF-MPS"
f = "04-BF-MPS"
# f = "05-LGU-MPS"
# f = "06-MGMEM-MPS-RR"

# Reading the log file
file = open(f'/home/ehyo/rad-scheduler/{f}/std.log', 'r')
Lines = file.readlines()

# Parsing the log file
for line in Lines:
    temp = line.split()
    ttemp = temp[0] + " " + temp[1]

    # Convert to datetime object
    date_format = '%d-%b-%y %H:%M:%S'
    time_point = datetime.strptime(ttemp, date_format)

    if temp[2] == "queued":
        if temp[3] not in job_queued_dispatched_dic:
            job_queued_dispatched_dic[temp[3]] = {'queued': [], 'dispatched': []}
        job_queued_dispatched_dic[temp[3]]['queued'].append(time_point)
    elif temp[2] == "dispatched":
        if temp[3] in job_queued_dispatched_dic:
            job_queued_dispatched_dic[temp[3]]['dispatched'].append(time_point)
        else:
            logging.warning(f"Dispatched event without queued event for task {temp[3]}. Adding dispatched only.")
            job_queued_dispatched_dic[temp[3]] = {'queued': [], 'dispatched': [time_point]}

# Calculating waiting times
waiting_time = pd.DataFrame(columns=['task_id', 'waiting_time(s)', 'waiting_time(m)', 'waiting_time(h)'])
for key in job_queued_dispatched_dic:
    queued = job_queued_dispatched_dic[key]['queued']
    dispatched = job_queued_dispatched_dic[key]['dispatched']
    if len(queued) > 0 and len(dispatched) > 0:
        total_waiting_time = sum([(dispatched[i] - queued[i]).total_seconds() for i in range(min(len(queued), len(dispatched)))])
        waiting_time_seconds = total_waiting_time
        waiting_time_minutes = waiting_time_seconds / 60
        waiting_time_hours = waiting_time_minutes / 60
        waiting_time.loc[len(waiting_time)] = [key, waiting_time_seconds, waiting_time_minutes, waiting_time_hours]
    else:
        logging.warning(f"Task {key} has incomplete queued or dispatched events.")

# Execution time section
exe_time_dic = dict()
list_of_files = []
for file in os.listdir(f"/home/ehyo/rad-scheduler/{f}"):
    if file.startswith("time") and file.endswith(".et"):
        file = os.path.join(f"/home/ehyo/rad-scheduler/{f}", file)
        list_of_files.append(file)

for iterator in list_of_files:
    file = open(f'{iterator}', 'r')
    Lines = file.readlines()

    pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    match = re.search(pattern, iterator)

    if match:
        task_id = match.group(0)
    else:
        logging.error(f"Failed to extract task ID from filename {iterator}. Skipping.")
        continue

    if len(Lines) != 0:
        for line in Lines:
            if "real" in line:
                a = line.strip().split()[1]
                minutes, rest = a.split("m")
                seconds = rest.split("s")[0]
                total_time_in_seconds = (float(minutes) * 60 + float(seconds))
                total_time_in_minutes = total_time_in_seconds / 60
                total_time_in_hours = total_time_in_minutes / 60

                if task_id in exe_time_dic.keys():
                    exe_time_dic[task_id][0] += total_time_in_seconds
                    exe_time_dic[task_id][1] += total_time_in_minutes
                    exe_time_dic[task_id][2] += total_time_in_hours
                else:
                    exe_time_dic[task_id] = [total_time_in_seconds, total_time_in_minutes, total_time_in_hours]
            else:
                continue
    else:
        logging.warning(f"No execution time data for task {task_id}. Setting execution time to 0.")
        exe_time_dic[task_id] = [0, 0, 0]

# Creating execution time DataFrame
exe_time = pd.DataFrame(columns=['task_id', 'execution_time(s)', 'execution_time(m)', 'execution_time(h)'])
for key in exe_time_dic:
    exe_time.loc[len(exe_time)] = [key, exe_time_dic[key][0], exe_time_dic[key][1], exe_time_dic[key][2]]

# Merging data
merged_df = waiting_time.merge(exe_time, on='task_id', how='outer')
merged_df.fillna(0, inplace=True)
merged_df["job_completion_time(s)"] = merged_df["execution_time(s)"] + merged_df["waiting_time(s)"]
merged_df["job_completion_time(m)"] = merged_df["execution_time(m)"] + merged_df["waiting_time(m)"]
merged_df["job_completion_time(h)"] = merged_df["execution_time(h)"] + merged_df["waiting_time(h)"]

# print(merged_df)
# merged_df.to_csv("summary.csv", index=False)

# Calculate averages over all metrics
average_metrics = {
    "average_waiting_time(s)": merged_df["waiting_time(s)"].mean(),
    "average_waiting_time(m)": merged_df["waiting_time(m)"].mean(),
    "average_waiting_time(h)": merged_df["waiting_time(h)"].mean(),
    "average_execution_time(s)": merged_df["execution_time(s)"].mean(),
    "average_execution_time(m)": merged_df["execution_time(m)"].mean(),
    "average_execution_time(h)": merged_df["execution_time(h)"].mean(),
    "average_job_completion_time(s)": merged_df["job_completion_time(s)"].mean(),
    "average_job_completion_time(m)": merged_df["job_completion_time(m)"].mean(),
    "average_job_completion_time(h)": merged_df["job_completion_time(h)"].mean(),
}

# Save average metrics to CSV in vertical format
avg_df = pd.DataFrame(average_metrics, index=["Value"]).T  # Transpose for vertical layout
avg_df.to_csv("summary_metrics.csv", header=False)

# Print averages for review
print("Average Metrics (Vertical):")
print(avg_df)

# Print averages for review
print("Average Metrics:")
print(avg_df)

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(merged_df['task_id'], merged_df['waiting_time(m)'], label="Waiting Time (minutes)", color="blue", alpha=0.6)
plt.bar(merged_df['task_id'], merged_df['execution_time(m)'], label="Execution Time (minutes)", bottom=merged_df['waiting_time(m)'], color="green", alpha=0.6)
plt.scatter(merged_df['task_id'], merged_df['job_completion_time(m)'], label="Job Completion Time (minutes)", color="red", lw=2)
plt.xticks(rotation=90, ha='right')
plt.legend()
plt.xlabel("Task IDs")
plt.ylabel("Time (minutes)")
plt.tight_layout()
plt.savefig("wait_exe_time.png")
