import socket
import time
import datetime
from threading import Thread, Lock
import subprocess
import os
import logging

import monitor
import rad_parser
from task_queue import Task, Tasks

from itertools import cycle, islice

import process_detector

# from xgboost import XGBRegressor
import pickle

# todos
# TODO: making parser function that gets the job content and gives out the command, etc. for the scheduler
# TODO: making executor function that gets the GPU and the command

# cnn_loaded_model = pickle.load(open("extraTree_cnn_mem.pickle.dat", "rb"))
# fc_loaded_model = pickle.load(open("extraTree_fc_mem.pickle.dat", "rb"))

# logger for keeping track of submission, dispatch, and termination time
logging.basicConfig(filename='std.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 

# locks for avoiding race condition
lock = Lock()
recover_lock = Lock()

# queues for submitted jobs, and
main_queue = Tasks()
recovery_queue = Tasks()


# List of GPU IDs
GPU_IDs = [
    "GPU-00f900e0-bb6f-792a-1b8a-597214c7e1a1",
    "GPU-36631a8a-069f-d2e7-5dbc-954ff1c64d8a",
    "GPU-2542cdb4-b558-5541-6feb-c4b72b612395",
    "GPU-02bed562-1235-134d-6e37-99f97fd3c1e0"
]

# Round-robin generator
round_robin_generator = cycle(GPU_IDs)

def select_ids(n):
    """
    Selects n IDs in a round-robin manner.
    Args:
        n (int): Number of IDs to select.
    Returns:
        list: List of selected IDs.
    """
    return list(islice(round_robin_generator, n))


handled_crashes = []

def recovery():
    print("recovery process started ...")
    time.sleep(5)
    """
        This is the function that checks error files and adds OOM found to the high-priority queue
    """
    # going through all of the error files and detecting OOM error and adding to the recovery queue
    # for the next phase of scheduling

    # TODO: making it more general to go through submitted jobs from different users directories
    list_of_files = []
    for file in os.listdir("/home/ehyo/rad-scheduler"):
        if file.startswith("err") and file.endswith(".log"):
            file = os.path.join("/home/ehyo/rad-scheduler", file)
            list_of_files.append(file)


    crashes = 0
    all_executions = 0
    for iterator in list_of_files:
        if iterator in handled_crashes:                 # if a task is recovered once, no need to be added again
                                                        # initial policy is to recover and give an idle A100 GPU
            continue
        else:
            all_executions += 1
            file = open(f'{iterator}', 'r')             # reading a file
            Lines = file.readlines()                    # reading the lines of that file

            for line in Lines: # going through lines of an opened file to find if the execution crashed due to OOM
                if "unsuccessful" in line:#"OOM" in line or "Non-OK-status" in line or "RESOURCE_EXHAUSTED" in line:
                    crashes += 1
                    
                    handled_crashes.append(iterator)    # We add it to the handled one to prevent over-scheduling
                    opener = open(f'{iterator}', 'r')   # We open the err file that has OOM 
                    Lines = opener.readlines()          # The goal is to fetch the information in the head of the err file

                    recovery_data = Lines[0].split('+')
                    # print(recovery_data)

                    tmp_dir = recovery_data[0]          # directory
                    tmp_file = recovery_data[3]         # rad file
                    tmp_user = recovery_data[4]         # user
                    tmp_task_id = recovery_data[5][:-1]      # task_id that was tokenized in the system

                    recovered_task = Task(tmp_user, tmp_dir, tmp_file)      # made the task object
                    recovered_task.set_id(tmp_task_id)                      # set the task_id to the initial one

                    recovered_task.set_if_recovered()
                    with recover_lock:
                        recovery_queue.enqueue(recovered_task)                  # putting the task in the recovery queue
                    print("OOM FOUND: recovery queue is filled with the task that has problem: ", recovered_task, recovered_task._to_string())
                    print("length of the queue:", recovery_queue.length())
                    break
    print("end of checking for failures ...")


def command_executor(command):
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')
    pass


def server():
    host = socket.gethostname()
    port = 5001

    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((host, port))

    while True:
        # configure how many client the server can listen simultaneously
        # It can be configured for any reason
        server_socket.listen(10)
        while True:
            conn, address = server_socket.accept()
            print("Connection from: " + str(address))
            data = conn.recv(1024).decode()

            if not data:
                break
            message = "Got your task and queued it."

            conn.send(message.encode())
            
            print(data)
            user, dir, file = data.split('+')
            file = file[1:]
            
            file = dir+"/"+file

            # print(user, dir, file)
            # it needs to be developed well :D

            a = Task(user, dir, file)

            with lock:
                main_queue.enqueue(a)

                logging.info(f"queued {a.task_id} - {a.file}")

        conn.close()



# this is the module implementing different policies for 
# assigning GPUs to tasks/ collocate tasks under different policies
def scheduler(policy = "oracle-most-GMem-available"):
    # exclusive, round-robin
    # oracle-first-fit, oracle-most-GMem-available, oracle-best-fit, oracle-least_GPU_utiltized
    # most-GMem-available-RR, least_GPU_utilized-RR, 
    # estimate-most-GMem-available (estimators needs to be set for the experiment first)

    # it needs to be a never-ending loops checking task queue, and GPUs, and making decisions
    estimator = "GPUMemNet"
    # horus
    # faketensor, GPUMemNet, oracle

    esIndex = 8

    if estimator == "horus":
        print("horus :)")
        esIndex = 9
    elif estimator == "faketensor":
        print("faketensor :)")
        esIndex = 10
    elif estimator == "GPUMemNet":
        print("GPUMemNet :)")
        esIndex = 11
    else:
        print("oracle :)")

    while True:
        recovery()

        # if there are some tasks waiting to be recovered/ served
        if main_queue.length() != 0 or recovery_queue.length() != 0:
            
            # here is the code body that implements exclusive policy
            # waiting as some tasks take time to start using GPU
            time.sleep(30)

            idle_gpus_to_send_job = list()
            gpus_activeness = monitor.gpus_activeness()
            for gpu in gpus_activeness:
                if gpus_activeness[gpu] == 0:
                    idle_gpus_to_send_job.append(gpu)
                    
            print("idle gpus: ", idle_gpus_to_send_job)
            
            # checking the job at the head of the recovery/ queue
            a = None
            main_queue_flag = None
            user, dir, file = None, None, None

            # Having higher priority for the tasks that need to be recovered
            if recovery_queue.length() != 0:
                with recover_lock:
                    a = recovery_queue.check()
                user, dir, file = a.user, a.dir, a.file
                main_queue_flag = False
            else:
                with lock:
                    a = main_queue.check()
                user, dir, file = a.user, a.dir, a.file
                main_queue_flag = True


            command = f"cd {dir} ; cat {file}"
            ret = subprocess.run(command, capture_output=True, shell=True)
            commands = ret.stdout.decode()
            commands_to_execute = commands.split("\n")

            # finding conda environment name
            env_name = None
            for command in commands_to_execute:
                if "activate" in command:
                    env_name = commands_to_execute[1].split("activate")[1].strip()
                    break

            if env_name == None:
                    env_name = "tf"

            # enabling the conda environment
            environment = f"/home/{user}/.conda/envs/{env_name}"
            print("conda environment to activate: ", env_name, environment)

            # finding the python code to execute 
            command_to_execute = None
            for command in commands_to_execute:
                if "python" in command:
                    command_to_execute = command
                    break
            if command_to_execute == None:
                print("the command could not be found in the submitted job profile!", file)

            number_of_GPUs_requested = int(commands_to_execute[7])

            print("command to execute: ", command_to_execute)
            print("number of gpus requested: ", number_of_GPUs_requested)

            now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

            if len(idle_gpus_to_send_job) >= number_of_GPUs_requested:
                assigned_gpus = idle_gpus_to_send_job[:number_of_GPUs_requested]

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                
                gpus_identifiers = ""
                for gpu in assigned_gpus:
                    if len(gpus_identifiers) > 0: 
                        gpus_identifiers += f",{gpu}"
                    else:
                        gpus_identifiers += f"{gpu}"

                logging.info(f"dispatched {a.task_id} - {a.file} - {gpus_identifiers}")

                print("assigned gpus with identifiers: ", gpus_identifiers)

                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""

                # discarding the task, as it has got the resources and got submitted for the execution
                if main_queue_flag == True:
                    main_queue.dequeue()
                else:
                    recovery_queue.dequeue()

                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'

                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()

                continue

            # ========================================================================================
            # ================================= ORACLE - FIRST FIT ===================================
            # ========================================================================================
            # elif policy == "oracle-first-fit" and recovery_queue.length() == 0:
            elif policy == "oracle-first-fit":
                print("First-Fit collocation")

                print("waiting for 60 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None
                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True


                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # gpu memory requirement
                gpu_memory_requirement = int(commands_to_execute[8])

                print("memory requirement: ", gpu_memory_requirement)
                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.Gmetrics
                

                # making sure that none of them are executing the multi-GPU tasks
                # the multi-gpu tasks are: xlnet, gpt2
                # we should take them out of the gpus that are gonna be candidate :)
                # multi_gpu_tasks_involved_gpus = process_detector.find_python_scripts()

                # if len(multi_gpu_tasks_involved_gpus) > 0:
                #     for gpu_uuids_to_remove in multi_gpu_tasks_involved_gpus:
                #         gpus_with_metrics = gpus_with_metrics.drop(index=gpu_uuids_to_remove, errors='ignore')


                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement
                # ===============
                temp_ = gpus_with_metrics.loc[(gpus_with_metrics['GPU_mem_available']) >= (gpu_memory_requirement + 2000)]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue

                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                assigned_gpus = candidate_gpus.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # /////////////////////////////////////////////////////////////////////////////////
                # /////////////////////////////////////////////////////////////////////////////////

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()
                
                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "oracle first fit")

                continue

            # ========================================================================================
            # ================================= ORACLE - BEST FIT ===================================
            # ========================================================================================
            # elif policy == "oracle-best-fit" and recovery_queue.length() == 0:
            elif policy == "oracle-best-fit":
                print("Best-Fit collocation")

                print("waiting for 30 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None
                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True
                        
                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)


                gpu_memory_requirement = int(commands_to_execute[8])

                # Finding the GPUs that the task can have :)
                gpus_with_metrics = monitor.Gmetrics
                
                # we can have these preconditions as well
                temp_ = gpus_with_metrics.loc[(gpus_with_metrics['GPU_mem_available']) >= (gpu_memory_requirement + 500)]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("No GPUs to submit job to!")
                    continue
                else:
                    print("The gpus that we can send job to :) \n", candidate_gpus)
                
                # sorting to be the best fit 
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=True, kind="mergesort")

                print("gpus sorted based on their available memory:\n", sorted)

                number_of_GPUs_requested = int(commands_to_execute[7])

                print("number of gpus requested: ", number_of_GPUs_requested)

                if number_of_GPUs_requested > len(sorted):
                    print("no available gpu, let's observe!")
                    continue
                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()
                
                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "oracle best fit!")

                continue

            # ========================================================================================
            # =========================== MOST GPU MEMORY AVAILABLE =============================
            # ========================================================================================
            # === This policy collocates knowing the GMem, relying on the recovery method ===========
            # =================== for OOMs due to memry fragmentation ==============================
            # ========================================================================================
            # elif policy == "oracle-most-GMem-available" :
            elif policy == "oracle-most-GMem-available" and recovery_queue.length() == 0:
                print("oracle most GMEM available")

                print("waiting for 30 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True

                        
                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # gpu memory requirement
                gpu_memory_requirement = int(commands_to_execute[8])

                print("memory requirement: ", gpu_memory_requirement)
                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.Gmetrics
                

                # making sure that none of them are executing the multi-GPU tasks
                # the multi-gpu tasks are: xlnet, gpt2
                # we should take them out of the gpus that are gonna be candidate :)
                # multi_gpu_tasks_involved_gpus = process_detector.find_python_scripts()

                # if len(multi_gpu_tasks_involved_gpus) > 0:
                #     for gpu_uuids_to_remove in multi_gpu_tasks_involved_gpus:
                #         gpus_with_metrics = gpus_with_metrics.drop(index=gpu_uuids_to_remove, errors='ignore')


                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement/ util
                # ===============
                temp_ = gpus_with_metrics.loc[gpus_with_metrics['GPU_mem_available'] >= (gpu_memory_requirement + 2000)]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue
                
                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)
                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # SORTING THE GPUS TO PRIORITIZE THE ONES WITH THE MOST GPU MEMORY
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # /////////////////////////////////////////////////////////////////////////////////
                # /////////////////////////////////////////////////////////////////////////////////

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory

                
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "ORACLE MOST GPU MEMORY AVAILABLE")

                continue







            # ==================================== Estimators ========================================
            # =========================== MOST GPU MEMORY AVAILABLE =============================
            # ========================================================================================
            # === This policy collocates based on the info it gets from the estimators ===========
            # ========================================================================================
            # elif policy == "estimate-most-GMem-available" and recovery_queue.length() == 0:
            elif policy == "estimate-most-GMem-available" and recovery_queue.length() == 0:
                print("most-gpu_memory_available")

                print("waiting for 30 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True

                        
                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # gpu memory requirement
                gpu_memory_requirement = float(commands_to_execute[esIndex])

                print("memory requirement: ", gpu_memory_requirement)
                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.Gmetrics
                

                # making sure that none of them are executing the multi-GPU tasks
                # the multi-gpu tasks are: xlnet, gpt2
                # we should take them out of the gpus that are gonna be candidate :)
                # multi_gpu_tasks_involved_gpus = process_detector.find_python_scripts()

                # if len(multi_gpu_tasks_involved_gpus) > 0:
                #     for gpu_uuids_to_remove in multi_gpu_tasks_involved_gpus:
                #         gpus_with_metrics = gpus_with_metrics.drop(index=gpu_uuids_to_remove, errors='ignore')


                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement/ util
                # ===============
                temp_ = gpus_with_metrics.loc[(gpus_with_metrics['GPU_mem_available']) >= (gpu_memory_requirement)]
                # candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                candidate_gpus = temp_

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue
                
                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)
                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # SORTING THE GPUS TO PRIORITIZE THE ONES WITH THE MOST GPU MEMORY
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # /////////////////////////////////////////////////////////////////////////////////
                # /////////////////////////////////////////////////////////////////////////////////

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory

                
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "ORACLE MOST GPU MEMORY AVAILABLE")

                continue


            
            # =========================================================================
            # ============== Collocation with knowing the memory requirement ==========
            # ================== Least utilized GPUs are prioritized ==================
            # =========================================================================

            # elif policy == "oracle-least_GPU_utiltized" and recovery_queue.length() == 0:
            elif policy == "oracle-least_GPU_utiltized" and recovery_queue.length() == 0:
                print("oracle least utilized GPU")

                print("waiting for 30 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True
                        
                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # gpu memory requirement
                gpu_memory_requirement = int(commands_to_execute[8])

                print("memory requirement: ", gpu_memory_requirement)

                # Finding the GPUs that the task can have :)
                gpus_with_metrics = monitor.Gmetrics
                
                # ==============================
                # ====== condition 1 for filtering the GPUs based on the GPU memory requirement/ util
                # ==============================
                temp_ = gpus_with_metrics.loc[(gpus_with_metrics['GPU_mem_available']) >= (gpu_memory_requirement + 2000)]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("No GPUs to submit job to!")
                    continue
                else:
                    print("The gpus that we can send job to :) \n", candidate_gpus)

                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # sorting the gpus based on their utilization in an ascending way
                sorted = candidate_gpus.sort_values(by="smact", kind="mergesort")
                print("gpus sorted based on their available memory:\n", sorted)    
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # /////////////////////////////////////////////////////////////////////////////////
                # /////////////////////////////////////////////////////////////////////////////////

                # writing logs to the system log
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'


                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "ORACLE least gpu utilized!")

                
                continue

            # ========================================================================================
            # =========================== MOST GPU MEMORY AVAILABLE =============================
            # ========================================================================================
            # === This policy collocates without knowing the GMem, relying on the recovery method =====
            # ================================ for OOMs  ==============================
            # ========================================================================================
            elif policy == "most-GMem-available-RR" and recovery_queue.length() == 0:
                print("most gpu memory available RR")

                print("waiting for 30 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True

                        
                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # gpu memory requirement
                # gpu_memory_requirement = int(commands_to_execute[8])

                # print("memory requirement: ", gpu_memory_requirement)
                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.Gmetrics
                

                # making sure that none of them are executing the multi-GPU tasks
                # the multi-gpu tasks are: xlnet, gpt2
                # we should take them out of the gpus that are gonna be candidate :)
                # multi_gpu_tasks_involved_gpus = process_detector.find_python_scripts()

                # if len(multi_gpu_tasks_involved_gpus) > 0:
                #     for gpu_uuids_to_remove in multi_gpu_tasks_involved_gpus:
                #         gpus_with_metrics = gpus_with_metrics.drop(index=gpu_uuids_to_remove, errors='ignore')


                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement/ util
                # ===============
                temp_ = gpus_with_metrics.loc[(gpus_with_metrics['GPU_mem_available']) >= 2000]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.80]

                # candidate_gpus = gpus_with_metrics

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue
                
                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)
                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # SORTING THE GPUS TO PRIORITIZE THE ONES WITH THE MOST GPU MEMORY
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # /////////////////////////////////////////////////////////////////////////////////
                # /////////////////////////////////////////////////////////////////////////////////

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # Generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory

                
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "MOST GPU MEMORY AVAILABLE relyaing on exclusive Recovery method")

                continue

            # ========================================================================================
            # =========================== Round Robin with Recovery ==================================
            # ========================================================================================
            elif policy == "round-robin" and recovery_queue.length() == 0:
                print("round robin")
                print("waiting for 30 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True

                        
                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                assigned_gpus = select_ids(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # /////////////////////////////////////////////////////////////////////////////////
                # /////////////////////////////////////////////////////////////////////////////////

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # Generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory

                
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "round robin relyaing on exclusive Recovery method")

                continue

            # =========================================================================
            # =========================================================================
            # ================== Least utilized GPUs are prioritized ==================
            # =========================================================================
            # =========================================================================

            elif policy == "least_GPU_utilized-RR" and recovery_queue.length() == 0:
                print("least utilized GPU")

                print("waiting for 30 seconds so the behavior of tasks can stabilize ...")
                time.sleep(30)

                a = None
                user, dir, file = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, file = a.user, a.dir, a.file
                    main_queue_flag = True
                        
                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # Finding the GPUs that the task can have :)
                gpus_with_metrics = monitor.Gmetrics
                
                # ==============================
                # ====== condition 1 for filtering the GPUs having at least 5 gig and utilized less than 80%
                # ==============================
                
                temp_ = gpus_with_metrics.loc[gpus_with_metrics['GPU_mem_available'] >= 2000]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("No GPUs to submit job to!")
                    continue
                else:
                    print("The gpus that we can send job to :) \n", candidate_gpus)


                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # sorting the gpus based on their utilization in an ascending way
                sorted = candidate_gpus.sort_values(by="smact", kind="mergesort")
                print("gpus sorted based on their available memory:\n", sorted)    
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # /////////////////////////////////////////////////////////////////////////////////
                # /////////////////////////////////////////////////////////////////////////////////

                # writing logs to the system log
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # generating the command that will execute
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; {{ time {command_to_execute} 1> out-{now}-{a.task_id}.log 2>> err-{now}-{a.task_id}.log ; }} 2> time-{now}-{a.task_id}.et & pid=$!
                    wait $pid
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{now}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{now}-{a.task_id}.log
                    fi"""
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{now}-{a.task_id}.log'


                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()
                
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "Least gpu utilized!")

                
                continue















            # ========================================================================================
            # =========================== LEAST UTILIZED GPU POLICY =============================
            # ========================================================================================
            elif policy == "least_utilized_GPU" and recovery_queue.length() == 0:
                print("we are here, least_utilized_GPU")

                time.sleep(60)
                gpus_with_metrics = monitor.Gmetrics
                # print("decision: \n", gpus_with_metrics)
                temp_ = gpus_with_metrics.loc[(gpus_with_metrics['GPU_mem_available']) >= 12000]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                # print("candidate GPUs:\n", candidate_gpus)

                sorted = candidate_gpus.sort_values(by="smact", kind="mergesort")

                print("gpus sorted:\n", sorted)

                if candidate_gpus.empty:
                    print("No GPUs to submit job to!")
                    continue
                else:
                    print("The gpus that we can send job to :) \n", candidate_gpus)
                    
                    # GPU selected here :)
                    candidate_gpu_to_collocate_job = sorted.index[0]

                    print("candidate GPU: ", candidate_gpu_to_collocate_job)

                    # sort and finding the actual one we want to target
                    a = None
                    user, dir, file = None, None, None
                    with lock:
                        a = main_queue.dequeue()
                        user, dir, file = a.user, a.dir, a.file
                        a.set_service_time(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
                        a.set_status("dispatched")

                    command = f"cd {dir} ; cat {file}"
                    ret = subprocess.run(command, capture_output=True, shell=True)
                    commands = ret.stdout.decode()
                    commands_to_execute = commands.split("\n")

                    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                    # finding conda environment name
                    env_name = None
                    for command in commands_to_execute:
                        if "activate" in command:
                            env_name = commands_to_execute[1].split("activate")[1].strip()
                            break
                    if env_name == None:
                            env_name = "tf"

                    # enabling the conda environment
                    environment = f"/home/{user}/.conda/envs/{env_name}"
                    print(env_name, environment)

                    # finding the python code to execute 
                    command_to_execute = None
                    for command in commands_to_execute:
                        if "python" in command:
                            command_to_execute = command
                            break
                    if command_to_execute == None:
                        print("the command could not be found in the submitted job profile!")

                    print("command to execute found: ", command_to_execute)

                    # writing logs to the system log
                    # writing logs to the system log
                    logging.info(f"dispatched {a.task_id} - {candidate_gpu_to_collocate_job}")

                    # generating the command that will execute
                    # command = f'cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={idle_gpu_to_send_job} ; {{ time {command_to_execute} 1> out-{user}-{now}-{file}-{a.task_id}.log 2> err-{user}-{now}-{file}-{a.task_id}.log ; }} 1> time1-{user}-{now}-{file}-{a.task_id}.et 2> time2-{a.task_id}.et || echo "fail" &'
                    command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={candidate_gpu_to_collocate_job} ; {{ time {command_to_execute} 1> out-{user}-{now}-{file}-{a.task_id}.log 2>> err-{user}-{now}-{file}-{a.task_id}.log ; }} 2> time-{user}-{now}-{file}-{a.task_id}.et & pid=$!
                        wait $pid 
                        if [ $? -eq 0 ]; then
                            echo 'Successful' >> err-{user}-{now}-{file}-{a.task_id}.log
                        else
                            echo 'unsuccessful' >>  err-{user}-{now}-{file}-{a.task_id}.log
                        fi
                        """
                    
                    to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{user}-{now}-{file}-{a.task_id}.log'

                    # print(command)
                    # print(to_write)

                    # subprocess.run(to_write, shell=True, check=True, executable='/bin/bash')
                    # subprocess.run(command, shell=True, check=True, executable='/bin/bash')
                    Thread(target = command_executor, args=(to_write,)).start()
                    Thread(target = command_executor, args=(command,)).start()

                # waiting for a while til the job goes on the GPU
                time.sleep(1)
                print("one allocation task is done!")

            # ========================================================================================
            # =========================== Policy with resource estimator =============================
            # ========================================================================================
            # === This policy collocates based on estimation, relying on the recovery method =========
            # ========================================================================================

            elif policy == "ml_predictor" and recovery_queue.length() == 0:
                # using my resource predictor based on the gathered dataset 

                a = None
                user, dir, file = None, None, None

                with lock:
                    a = main_queue.dequeue()
                    user, dir, file = a.user, a.dir, a.file
                    a.set_service_time(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
                    a.set_status("dispatched")

                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print(env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # sending the info for the parser to give out the feature for the estimator
                # file = "model.txt", dataset = "/raid/datasets/imagenet", batch_size = 32
                cnn_features, fc_features, overhead = rad_parser.analyze_model_summary(f"{dir}/{commands_to_execute[3]}", commands_to_execute[4], int(commands_to_execute[5]))

                # loading the memory consumption predictor :)
                global cnn_loaded_model
                global fc_loaded_model

                cnn_memory_predictor = cnn_loaded_model
                fc_memory_predictor = fc_loaded_model

                cnn_predicted_memory = cnn_memory_predictor.predict(cnn_features)
                fc_predicted_memory = fc_memory_predictor.predict(fc_features)

                print(cnn_predicted_memory, fc_predicted_memory, overhead)

                all_memory_estimation = cnn_predicted_memory[0] + fc_predicted_memory[0] + overhead
                # ============== getting the list of metrics and detecting GPUs that can be candidates for sending more tasks =========
                time.sleep(61)
                gpus_with_metrics = monitor.Gmetrics
                # print("decision: \n", gpus_with_metrics)
                temp_ = gpus_with_metrics.loc[gpus_with_metrics['GPU_mem_available'] > (all_memory_estimation)]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                # candidate_gpus = temp_
                
                # print("candidate GPUs:\n", candidate_gpus)

                # by="GPU_mem_available", ascending=False, kind="mergesort"
                # sorted = candidate_gpus.sort_values(by="smact", kind="mergesort")
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")

                print("gpus sorted:\n", sorted)

                if candidate_gpus.empty:
                    print("No GPUs to submit job to!")
                    with lock:
                        main_queue.put_it_back(a)
                    continue
                else:
                    print("The gpus that we can send job to :) \n", candidate_gpus)
                    
                    # GPU selected here :)
                    candidate_gpu_to_collocate_job = sorted.index[0]

                    print("candidate GPU: ", candidate_gpu_to_collocate_job)
                # ====================================================================

                # TODO: looking for the list of the GPUs to find the first-fit
                # TODO: it can also be best-fit

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {candidate_gpu_to_collocate_job}")

                # generating the command that will execute
                # command = f'cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={idle_gpu_to_send_job} ; {{ time {command_to_execute} 1> out-{user}-{now}-{file}-{a.task_id}.log 2> err-{user}-{now}-{file}-{a.task_id}.log ; }} 1> time1-{user}-{now}-{file}-{a.task_id}.et 2> time2-{a.task_id}.et || echo "fail" &'
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={candidate_gpu_to_collocate_job} ; {{ time {command_to_execute} 1> out-{user}-{now}-{file}-{a.task_id}.log 2>> err-{user}-{now}-{file}-{a.task_id}.log ; }} 2> time-{user}-{now}-{file}-{a.task_id}.et & pid=$!
                    wait $pid 
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{user}-{now}-{file}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{user}-{now}-{file}-{a.task_id}.log
                    fi
                    """
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{user}-{now}-{file}-{a.task_id}.log'

                # print(command)
                # print(to_write)

                # subprocess.run(to_write, shell=True, check=True, executable='/bin/bash')
                # subprocess.run(command, shell=True, check=True, executable='/bin/bash')
                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()

            timepoint = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print(timepoint, "Number of tasks waiting in the queue: ", main_queue.length())           
        else:
            # no task in the waiting queue
            # print("No task in the queue!")
            pass
    
    # time to check if there has been any crashes to be handled
    

header_flag_decision = True
def system_use_utilization_logger():
    while True:
        # waiting to the size of monitoring window
        time.sleep(60)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        gpus_with_metrics = monitor.Gmetrics
        tmp = gpus_with_metrics.assign(time=[now]*len(gpus_with_metrics))
        global header_flag_decision
        if header_flag_decision == True:
            tmp.to_csv('decision_making_monitored_data.csv', mode='a')
            header_flag_decision = False
        else:
            tmp.to_csv('decision_making_monitored_data.csv', mode='a', header = False)

header_flag_top = True
def top_system_logger():
    while True:
        time.sleep(3)
        a = monitor.top_extractor()

        global header_flag_top
        if header_flag_top == True:
            a.to_csv('top_data.csv', mode='a', index = False)
            header_flag_top = False
        else:
            a.to_csv('top_data.csv', mode='a', header = False, index=False)

if __name__ == '__main__':
    Thread(target = server).start()
    Thread(target = scheduler).start()
    Thread(target = monitor.metrics).start()
    Thread(target = system_use_utilization_logger).start()
    Thread(target = top_system_logger).start()