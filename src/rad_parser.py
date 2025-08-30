import os
from monitor import execute_command
import pandas as pd
from statistics import mean 


def dataset_analysis(dir = "/raid/datasets/imagenet/"):
    # finding the size, unit of data points in the directory of the dataset
    files_subfolders_size = execute_command(f"du -h --max-depth=1 {dir}")

    files_subfolders_size = files_subfolders_size.split("\n")

    total_size = 0
    unit = "G"

    for n in files_subfolders_size:
        tmp = n.split()
        if len(tmp) == 0:
            continue
        
        directory_to_check = tmp[1].replace(dir, "")
        if len(directory_to_check) == 0:
            continue

        size_with_unit = n.split()[0]
        
        temp = 0
        unit = size_with_unit[-1]
        size_value = size_with_unit[:-1]

        #  checking the unit and doing the divisions before adding them up
        if unit == "G":
            temp = float(size_value)
        elif unit == "M":
            temp = float(size_value)/ 1000
        elif unit == "K":
            temp = float(size_value)/ 1000000

        total_size += temp  # added to the overall size of the dataset

    # finding the number of datapoints
    command = f'cd {dir}; find -maxdepth 1 -type d | sort | while read -r dir; do n=$(find "$dir" -type f | wc -l); printf "%4d : %s\n" $n "$dir"; done' 
    files_count = os.popen(command).read()

    files_count = files_count.split("\n")
    total_count = 0
    for n in files_count:
        if n.split()[2] == '.':
            total_count = n.split()[0]
            break

    return total_size, unit, total_count

 




def analyze_model_summary(file = "vgg.txt", dataset = "/raid/datasets/imagenet", batch_size = 128):
    model = execute_command(f"cat {file}")
    model = model.split("\n")

    dataset_size_GB = 0 # DONE
    image_dimensions = 0 # we find it here from the input layer
    image_channels = 0 # we find it here from the input layer

    fc_neurons = 0
    fc_layers = 0
    fc_params = 0

    batch_size = batch_size
    total_params = 0
    trainable_params = 0
    convolutional_layers = 0 # we count is here
    activations = 0

    skip_conncetions_overhead = 0


    i = 0
    lines_to_ignore = {0, 1, 2, 3}
    for line in model:
        print("LINE: ", line)
        if (i in lines_to_ignore) or line.startswith("  ") or line.startswith("=") or line.startswith("_") or len(line) == 0:
            i += 1
            continue
        
        i += 1
        # getting input image dimensions from here
        if "input" in line and not "'input" in line:
            temp = line
            image_dimensions, _ , image_channels = [int(i) for i in temp.strip().split("input")[1].split("[(")[1].split(")]")[0].split(",")[1:]]
            print("found input", line, "\nextracted: ", int(image_dimensions), int(image_channels))
            

        elif "conv" in line and not "'conv" in line:
            line_listed = line.split()
            conv_params = line.strip().split()[6]
            conv_params = int(conv_params)
            convolutional_layers += 1
            activations += (int(line_listed[3][:-1]) * int(line_listed[4][:-1]) * int(line_listed[5][:-1]))
        
        elif "batch_normalization" in line and not "'batch_nor" in line:
            line_listed = line.split()
            activations += (int(line_listed[3][:-1]) * int(line_listed[4][:-1]) * int(line_listed[5][:-1]))

        elif ("max_pool" in line and not "\'max_pool" in line) or ("global_average_pooling" in line and not "\'global_average_pooling" in line):
            line_listed = line.split()
            print(line_listed)
            activations += int(line_listed[3][:-1])

        elif "dense" in line and not "'dense" in line:
            line_listed = line.split()
            # print(line_listed[3][:-1])
            # exit()
            neurons = int(line_listed[3][:-1])
            fc_neurons += neurons
            fc_layers += 1
            activations += neurons
            fc_params += int(line_listed[4])

        elif "re_lu" in line:
            line_listed = line.split()
            print(line_listed)
            activations += (int(line_listed[3][:-1]) * int(line_listed[4][:-1]) * int(line_listed[5][:-1]))

        elif "flatten" in line:
            activations += int((line.strip().split()[3])[:-1])
             

        elif "Trainable params:" in line:
            trainable_params += int((line.split("Trainable params:")[1].strip()).split("(")[0].strip())


        elif "Total params:" in line:
            total_params = int((line.split("Total params:")[1].strip()).split("(")[0].strip())


        elif ("concatenate" in line and not "'concatena" in line) or "add" in line:
            line_listed = line.split()
            print(line_listed)
            skip_conncetions_overhead += int(line_listed[3][:-1]) * int(line_listed[4][:-1]) * int(line_listed[5][:-1])

        else:
            pass
            print("ignored", line, i)


    # if "mnist" in dataset:
    #     dataset_size_GB = 0.023
    #     image_dimensions = 28
    #     image_channels = 1
    # else:
    #     size_value, unit, _ = dataset_analysis(dir = dataset)

    #     print(size_value, unit)
    #     if unit == "K":
    #         dataset_size_GB = size_value / 1000000
    #         pass
    #     elif unit == "M":
    #         dataset_size_GB = size_value / 1000
    #         pass
    #     elif unit == "G":
    #         # do nothing
    #         dataset_size_GB = size_value



    overhead = (skip_conncetions_overhead * batch_size * 2) * 4 / 2**20
    print(skip_conncetions_overhead, overhead)
    # activations += skip_conncetions_overhead
    cnn_features = [[convolutional_layers, total_params, activations, batch_size]]

    fc_features = [[fc_layers, fc_params, fc_neurons, batch_size]]
    

    return cnn_features, fc_features, overhead





# import pickle
# cnn_loaded_model = pickle.load(open("extraTree_cnn_mem.pickle.dat", "rb"))
# a = analyze_model_summary(file = "vgg.txt", dataset = "/raid/datasets/imagenet", batch_size = 128)

# print(a, cnn_loaded_model.predict(a))






# ========================================================================
# ====================== old testing =====================================
# ================ with the ML models for memory prediction ==============
# ========================================================================

# import subprocess
# user = "ehyo"
# dir = "/home/ehyo/rad-scheduler"
# file1 = open('scenario.sh', 'r')

# for a in file1:
#     if "sleep" in a:
#         continue
#     tmp_ = a.split()
#     file = tmp_[3]

#     file = "densenet.rad"
#     print(file)
#     # exit()
#     command = f"cat {file}"
#     ret = subprocess.run(command, capture_output=True, shell=True)
#     commands = ret.stdout.decode()
#     commands_to_execute = commands.split("\n")

#     # print(commands, commands_to_execute)
#     # exit()
#     # finding conda environment name
#     env_name = None
#     for command in commands_to_execute:
#         if "activate" in command:
#             env_name = commands_to_execute[1].split("activate")[1].strip()
#             break
#     if env_name == None:
#             env_name = "tf"

#     # enabling the conda environment
#     environment = f"/home/{user}/.conda/envs/{env_name}"
#     print(env_name, environment)

#     # finding the python code to execute 
#     command_to_execute = None
#     for command in commands_to_execute:
#         if "python" in command:
#             command_to_execute = command
#             break
#     if command_to_execute == None:
#         print("the command could not be found in the submitted job profile!")

#     print("command to execute found: ", command_to_execute)



#     # print(f"{dir}/{commands_to_execute[3]}", commands_to_execute[4], int(commands_to_execute[5]))
#     # exit()
#     a, b = analyze_model_summary(f"{dir}/{commands_to_execute[3]}", commands_to_execute[4], int(commands_to_execute[5]))
#     print(a, b)
#     # exit()
#     cnn_predicted_memory = cnn_memory_predictor.predict(a)
#     # fc_predicted_memory = fc_memory_predictor.predict(b)

#     print(cnn_predicted_memory)
#     exit()
#     print(dataset_analysis())