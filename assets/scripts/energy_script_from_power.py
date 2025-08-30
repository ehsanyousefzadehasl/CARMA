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

# Initial_results_with_random_trace_scenario/07-new_exclusive
# Initial_results_with_random_trace_scenario/11-most_GPU_mem_4GB_with_recovery
@click.command()
# @click.option('--file', '--f', type=click.STRING, default="/home/ehyo/rad-scheduler/001-exclusive/dcgmi_metrics.csv", help='csv file address')
# @click.option('--file', '--f', type=click.STRING, default="/home/ehyo/rad-scheduler/002-oracle-first-fit/dcgmi_metrics.csv", help='csv file address')
# @click.option('--file', '--f', type=click.STRING, default="/home/ehyo/rad-scheduler/002-oracle-first-fit-balance-MPS/dcgmi_metrics.csv", help='csv file address')
@click.option('--file', '--f', type=click.STRING, default="/home/ehyo/rad-scheduler/001-exclusive-balance/dcgmi_metrics.csv", help='csv file address')
def main(file):
    # dcgmi log visualizer

    # df = pd.read_csv("dcgmi_metrics.csv")
    df = pd.read_csv(f"{file}")
    GPU_0 = df.loc[(df['gpu_id'] == 0)]
    GPU_1 = df.loc[(df['gpu_id'] == 1)]
    GPU_2 = df.loc[(df['gpu_id'] == 2)]
    GPU_4 = df.loc[(df['gpu_id'] == 4)]


    pow1 = GPU_0["power"]
    pow2 = GPU_1["power"]
    pow3 = GPU_2["power"]
    pow4 = GPU_4["power"]
    

    energy_GPU1, _ = calculate_energy_consumption(pow1)
    energy_GPU2, _ = calculate_energy_consumption(pow2)
    energy_GPU3, _ = calculate_energy_consumption(pow3)
    energy_GPU4, _ = calculate_energy_consumption(pow4)

    print("Energy: ", energy_GPU1 + energy_GPU2 + energy_GPU3 + energy_GPU4)



if __name__ == '__main__':
    main()