import pandas as pd

import click

@click.command()
@click.option('--file1', '--f', type=click.STRING, default="dcgmi_metrics.csv", help='dcgmi csv file address')
@click.option('--file2', '--f', type=click.STRING, default="top.csv", help='top csv file address')
# @click.option('--metric', '--m', type=click.STRING, default="gpu_memory_usage", help='enter the metric you want to be visualized')
def main(file1, file2):
    # dcgmi log visualizer

    df = pd.read_csv(f"{file1}")
    dff = pd.read_csv(f"{file2}")
    # GPU_0 = df.loc[(df['gpu_id'] == 0)]
    # GPU_1 = df.loc[(df['gpu_id'] == 1)]
    # GPU_2 = df.loc[(df['gpu_id'] == 2)]
    # GPU_4 = df.loc[(df['gpu_id'] == 4)]
    
    gpu_memory = df['gpu_memory_usage']/ 1000
    smact = df['smact']
    cpu_util = dff['%CPU']
    sys_memory = dff['used_memory']

    print("avg gpu memory: ", gpu_memory.mean(), "\n smact: ", smact.mean(), "\n cpu utilization: ", cpu_util.mean(), "\n system memory: ", sys_memory.mean())
    # print(data.sum(), data.mean(), len(data))
    exit()

    if metric == "gpu_memory_usage":
        data_to_visualize1 = GPU_0[f"{metric}"]/ 1000
        data_to_visualize2 = GPU_1[f"{metric}"]/ 1000
        data_to_visualize3 = GPU_2[f"{metric}"]/ 1000
        data_to_visualize4 = GPU_4[f"{metric}"]/ 1000
    else:
        data_to_visualize1 = GPU_0[f"{metric}"]
        data_to_visualize2 = GPU_1[f"{metric}"]
        data_to_visualize3 = GPU_2[f"{metric}"]
        data_to_visualize4 = GPU_4[f"{metric}"]

    
    print(data_to_visualize1.mean(), data_to_visualize2.mean(), data_to_visualize3.mean(), data_to_visualize4.mean())



    # df = pd.read_csv("top_data.csv")
    # used_memory = df["used_memory"]
    # CPU_UTIL = df["%CPU"]


if __name__ == '__main__':
    main()