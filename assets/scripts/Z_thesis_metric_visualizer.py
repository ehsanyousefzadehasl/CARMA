import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(directory, metric):
    """Loads and extracts GPU-specific metric data from the CSV file."""
    file = os.path.join(directory, "dcgmi_metrics.csv")
    df = pd.read_csv(file)

    # Extract data for GPUs 0, 1, 2, 4
    return {
        0: df.loc[df['gpu_id'] == 0, metric],
        1: df.loc[df['gpu_id'] == 1, metric],
        2: df.loc[df['gpu_id'] == 2, metric],
        4: df.loc[df['gpu_id'] == 4, metric]
    }

def plot_gpu_metrics(data1, data2, dir1, dir2, metric, smooth):
    """Plots the given metric for all GPUs with separate subplots for all GPUs."""
    scale_dict = {
        "gpu_memory_usage": 1000,
        "power": 1,
        "memory_copy": 1,
        "pci": 1e9,
        "nvl": 1e10,
        "smact": 1
    }
    
    # Use provided scale or default to 1 if metric is unknown
    scale = scale_dict.get(metric, 1)
    
    # Smoothing window size
    window_size = 40  

    # Consistent grayscale styles with lighter and darker versions
    styles = [
        {'light': '0.6', 'dark': '0.0'},
        {'light': '0.6', 'dark': '0.0'},
        {'light': '0.6', 'dark': '0.0'},
        {'light': '0.6', 'dark': '0.0'}
    ]
    gpu_ids = [0, 1, 2, 4]

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14

    fig, axs = plt.subplots(len(gpu_ids), 1, figsize=(12, 10), sharex=True)

    energy_overall1 = 0
    energy_overall2 = 0
    for i, gpu in enumerate(gpu_ids):
        

        data1_scaled = data1[gpu] / scale
        data2_scaled = data2[gpu] / scale


        # print(f"GPU {gpu} energy: ", data1[gpu], data1[gpu].iloc[0], data1[gpu].iloc[-1])
        if metric == "energy":
            print(data1[gpu].iloc[-1] - data1[gpu].iloc[0])
            energy_overall1 += (data1[gpu].iloc[-1] - data1[gpu].iloc[0])/(10**9)
            energy_overall2 += (data2[gpu].iloc[-1] - data2[gpu].iloc[0])/(10**9)
            print("energy: ", energy_overall1, energy_overall2)

        # print(data1_scaled)
        # exit()

        # Apply smoothing if enabled
        if smooth:
            data1_scaled = data1_scaled.rolling(window=window_size, center=True).mean()
            data2_scaled = data2_scaled.rolling(window=window_size, center=True).mean()

        # Solid line with lighter color for first dataset

        if "exclusive" in dir1:
            leg1 = "Exclusive"
        axs[i].plot(data1_scaled, label=f"{leg1}", 
                    color=styles[i]['light'], 
                    linestyle='-', 
                    linewidth=1.5)
        leg3  = ""
        if "MGMEM" in dir2:
            leg2 = "MAGM"
        elif "round" in dir2:
            leg2 = "Round Robin"
        elif "LGU" in dir2:
            leg2 = "LUG"
        elif "horus" in dir2:
            leg2 = "Horus"
        elif "fake" in dir2:
            leg2 = "FakeTensor"
        elif "GPUMemNet" in dir2:
            leg2 = "MAGM"
            leg3 = "GPUMemNet"
        elif "3GPU-MAGM" in dir2:
            leg2 = "3GPU-MAGM"

        # Dashed line with darker color for second dataset
        axs[i].plot(data2_scaled, label=f"{leg2}+{leg3}", 
                    color=styles[i]['dark'], 
                    linestyle='--', 
                    linewidth=2)
        
        if metric == "smact":
            unit = " (%)"
        elif metric == "power":
            unit = " (W)"
        elif metric == "gpu_memory_usage":
            unit = " (GB)"
        elif metric == "energy":
            unit = " (MJ)"

        if metric == "smact":
            axs[i].set_ylabel(f"{metric.upper()}{unit} - (GPU{gpu})")
        elif metric == "gpu_memory_usage":
            metric = "GPUMem"
            axs[i].set_ylabel(f"{metric.upper()}{unit} - (GPU{gpu})")
        else:
            axs[i].set_ylabel(f"{metric.upper()}{unit} - (GPU{gpu})")

        if metric == "power":
            axs[i].legend(loc='lower right', frameon=False)
        else:
            axs[i].legend(loc='upper right', frameon=False)
        axs[i].set_ylim(0, 40 if metric == "gpu_memory_usage" else None)
        axs[i].grid(True, linestyle='--', linewidth=0.5, color='0.85')

    plt.xlabel("Time (3-second intervals)")
    # plt.suptitle(f"Comparison of {metric} for All GPUs ({dir1} vs {dir2})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"{metric}_comparison_all_gpus{'_smoothed' if smooth else ''}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare GPU metrics from two directories.")
    parser.add_argument("--dir1", type=str, required=True, help="First directory containing the CSV file.")
    parser.add_argument("--dir2", type=str, required=True, help="Second directory containing the CSV file.")
    parser.add_argument("--metric", type=str, help="Metric to visualize (e.g., gpu_memory_usage, power, smact)")
    parser.add_argument("--smooth", action="store_true", help="Enable smoothing using a moving average")

    args = parser.parse_args()

    # Load data from both directories
    data1 = load_data(args.dir1, args.metric)
    data2 = load_data(args.dir2, args.metric)



    # Plot the comparison
    plot_gpu_metrics(data1, data2, args.dir1, args.dir2, args.metric, args.smooth)