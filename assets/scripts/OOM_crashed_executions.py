import os

list_of_files = [] # 05-least_GPU_utilized  04-most_GPU_mem_available
for file in os.listdir("/home/ehyo/rad-scheduler/10-Horus-MGMEM-80%smact"):
    if file.startswith("err") and file.endswith(".log"):
        file = os.path.join("/home/ehyo/rad-scheduler/10-Horus-MGMEM-80%smact", file)
        list_of_files.append(file)



crashes_due2_OOM = 0
all_executions = 0
for iterator in list_of_files:
    all_executions += 1
    file = open(f'{iterator}', 'r')
    Lines = file.readlines()

    for line in Lines:
        if "unsuccessful" in line:
            crashes_due2_OOM += 1
            break

    # for line in Lines:
    #     if "accuracy" in line:
    #         crashes_due2_OOM += 1
    #         break


print(crashes_due2_OOM, all_executions)