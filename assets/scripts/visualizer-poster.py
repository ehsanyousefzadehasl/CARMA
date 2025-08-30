import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe

import click


def calculate_energy_consumption(power_data, sampling_interval=3):
    """
    Calculate the total energy consumed from power data points.

    Args:
        power_data (list of float): List of power values in watts.
        sampling_interval (float): Time interval between each sample in seconds.

    Returns:
        energy_joules (float): Total energy consumed in joules.
        energy_wh (float): Total energy consumed in watt-hours.
    """
    # Calculate the total energy in joules
    energy_joules = sum(power * sampling_interval for power in power_data)

    # Convert energy to watt-hours (1 Wh = 3600 J)
    energy_wh = energy_joules / 3600

    return energy_joules, energy_wh

# Example power data points in watts (sampled every 3 seconds)
# power_data = [200, 220, 210, 205, 215, 225, 210]  # Example data

# # Calculate the energy consumption
# energy_joules, energy_wh = calculate_energy_consumption(power_data)

# print(f"Total Energy Consumed: {energy_joules:.2f} J")
# print(f"Total Energy Consumed: {energy_wh:.4f} Wh")



@click.command()
@click.option('--file', '--f', type=click.STRING, default="../03-MGMEM-MPS/dcgmi_metrics.csv", help='csv file address')
@click.option('--metric', '--m', type=click.STRING, default="gpu_memory_usage", help='enter the metric you want to be visualized')
def main(file, metric):
    # dcgmi log visualizer

    # df = pd.read_csv("dcgmi_metrics.csv")
    df = pd.read_csv(f"{file}")
    GPU_0 = df.loc[(df['gpu_id'] == 0)]
    GPU_1 = df.loc[(df['gpu_id'] == 1)]
    GPU_2 = df.loc[(df['gpu_id'] == 2)]
    GPU_4 = df.loc[(df['gpu_id'] == 4)]

    # print(GPU_1)

    # pow1 = GPU_0["power"]/ 1000
    # pow2 = GPU_1["power"]/ 1000
    # pow3 = GPU_2["power"]/ 1000
    # pow4 = GPU_4["power"]/ 1000
    

    # energy_GPU1, _ = calculate_energy_consumption(pow1)
    # energy_GPU2, _ = calculate_energy_consumption(pow1)
    # energy_GPU3, _ = calculate_energy_consumption(pow1)
    # energy_GPU4, _ = calculate_energy_consumption(pow1)

    # print("Energy: ", energy_GPU1 + energy_GPU2 + energy_GPU3 + energy_GPU4)
    scale = 0
    data_to_visualize1 = pd.DataFrame()
    data_to_visualize2 = pd.DataFrame()
    data_to_visualize3 = pd.DataFrame()
    data_to_visualize4 = pd.DataFrame()

    if metric == "gpu_memory_usage":
        data_to_visualize1 = GPU_0[f"{metric}"]/ 1000
        data_to_visualize2 = GPU_1[f"{metric}"]/ 1000
        data_to_visualize3 = GPU_2[f"{metric}"]/ 1000
        data_to_visualize4 = GPU_4[f"{metric}"]/ 1000
        scale = 40
    elif metric == "power":
        data_to_visualize1 = GPU_0[f"{metric}"]
        data_to_visualize2 = GPU_1[f"{metric}"]
        data_to_visualize3 = GPU_2[f"{metric}"]
        data_to_visualize4 = GPU_4[f"{metric}"]
        scale = 300
    elif metric == "memory_copy":
        data_to_visualize1 = GPU_0[f"{metric}"]
        data_to_visualize2 = GPU_1[f"{metric}"]
        data_to_visualize3 = GPU_2[f"{metric}"]
        data_to_visualize4 = GPU_4[f"{metric}"]
        scale = 100
    elif "pci" in metric:
        data_to_visualize1 = GPU_0[f"{metric}"]
        data_to_visualize2 = GPU_1[f"{metric}"]
        data_to_visualize3 = GPU_2[f"{metric}"]
        data_to_visualize4 = GPU_4[f"{metric}"]
        scale = 1000000000
    elif "nvl" in metric:
        data_to_visualize1 = GPU_0[f"{metric}"]
        data_to_visualize2 = GPU_1[f"{metric}"]
        data_to_visualize3 = GPU_2[f"{metric}"]
        data_to_visualize4 = GPU_4[f"{metric}"]
        scale = 10000000000
    else:
        data_to_visualize1 = GPU_0[f"{metric}"]
        data_to_visualize2 = GPU_1[f"{metric}"]
        data_to_visualize3 = GPU_2[f"{metric}"]
        data_to_visualize4 = GPU_4[f"{metric}"]
        scale = 1

    
    print(data_to_visualize1.mean(), data_to_visualize2.mean(), data_to_visualize3.mean(), data_to_visualize4.mean())
    # print("data average: ", (data_to_visualize1+data_to_visualize2+data_to_visualize3+data_to_visualize4)/4)
    plt.figure(figsize=(12,4))
    plt.plot(data_to_visualize1, label="GPU0", color='k', lw=2, path_effects=[pe.Stroke(linewidth=5, foreground='g'), pe.Normal()])
    plt.plot(data_to_visualize2, label="GPU1", linewidth = '3')
    plt.plot(data_to_visualize3, label="GPU2", linewidth = '1')
    plt.plot(data_to_visualize4, label="GPU4", linewidth = '2')
    flag = True

    plt.ylabel(f'{metric}')
    plt.xlabel('time (3 second intervals)')
    if metric == "gpu_memory_usage":
        plt.ylim((0,40))
    else:
        pass
    plt.legend()
    if flag == True:
        plt.savefig(f"{metric}-all-GPUs.png")
    else:
        plt.savefig(f"{metric}.png")


    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axs[0].plot(data_to_visualize1, color='r', linewidth=2)
    axs[0].set_ylabel(f'{metric} (GPU0)')
    axs[0].set_ylim(0, scale)

    axs[1].plot(data_to_visualize2, color='b', linewidth=2)
    axs[1].set_ylabel(f'{metric} (GPU1)')
    axs[1].set_ylim(0, scale)

    axs[2].plot(data_to_visualize3, color='g', linewidth=2)
    axs[2].set_ylabel(f'{metric} (GPU2)')
    axs[2].set_ylim(0, scale)

    axs[3].plot(data_to_visualize4, color='m', linewidth=2)
    axs[3].set_ylabel(f'{metric} (GPU3)')
    axs[3].set_ylim(0, scale)

    plt.xlabel('time (3 second intervals)')
    plt.tight_layout()
    plt.savefig(f"{metric}-subplots_raw_data.png")


    # with smoothing
    # Smoothing data using a moving average with a window size of 20
    window_size = 40

    smooth_data1 = pd.Series(data_to_visualize1).rolling(window=window_size, center=True).mean()
    smooth_data2 = pd.Series(data_to_visualize2).rolling(window=window_size, center=True).mean()
    smooth_data3 = pd.Series(data_to_visualize3).rolling(window=window_size, center=True).mean()
    smooth_data4 = pd.Series(data_to_visualize4).rolling(window=window_size, center=True).mean()

    # Plotting the smoothed data in subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axs[0].plot(smooth_data1, color='r', linewidth=2)
    axs[0].set_ylabel(f'{metric} (GPU0)')
    axs[0].set_ylim(0, scale)

    axs[1].plot(smooth_data2, color='b', linewidth=2)
    axs[1].set_ylabel(f'{metric} (GPU1)')
    axs[1].set_ylim(0, scale)

    axs[2].plot(smooth_data3, color='g', linewidth=2)
    axs[2].set_ylabel(f'{metric} (GPU2)')
    axs[2].set_ylim(0, scale)

    axs[3].plot(smooth_data4, color='m', linewidth=2)
    axs[3].set_ylabel(f'{metric} (GPU3)')
    axs[3].set_ylim(0, scale)

    plt.xlabel('time (3 second intervals)')
    plt.tight_layout()
    plt.savefig(f"{metric}_smoothed_subfigs.png")


# top data part
    # df = pd.read_csv("../001-exclusive/top_data.csv")
    # used_memory = df["used_memory"]
    # CPU_UTIL = df["%CPU"]

    # plt.figure(figsize=(12,4))
    # plt.plot(used_memory, label="Used System Memory")
    # plt.ylabel('Used System Memory (GB)')
    # plt.xlabel('time (3 second intervals)')
    # plt.ylim((0,150))
    # plt.legend()
    # plt.savefig("used_system_memory.png")

    # plt.figure(figsize=(12,4))
    # plt.plot(CPU_UTIL, label="CPU%")
    # plt.ylabel('CPU Utilization')
    # plt.xlabel('time (3 second intervals)')
    # plt.ylim((0,6400))
    # plt.legend()
    # plt.savefig("cpu_utilization.png")

if __name__ == '__main__':
    main()