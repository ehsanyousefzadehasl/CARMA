import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def label_dir(d):
    if "exclusive" in d:
        return "Exclusive"
    if "MGMEM" in d:
        return "MAGM"
    if "round" in d:
        return "Round Robin"
    if "LGU" in d:
        return "LUG"
    if "horus" in d:
        return "Horus"
    if "fake" in d:
        return "FakeTensor"
    if "GPUMemNet" in d:
        return "MAGM+GPUMemNet"
    if "3GPU-MAGM" in d:
        return "3GPU-MAGM"
    return os.path.basename(d)


def plot_subfigures_for_one_gpu(data1_dir, data2_dir, gpu_id, metrics, smooth):
    """Plot 3 vertically stacked subplots for different metrics of a single GPU."""
    scale_dict = {
        "gpu_memory_usage": 1000,
        "power": 1,
        "memory_copy": 1,
        "pci": 1e9,
        "nvl": 1e10,
        "smact": 1
    }

    df1 = pd.read_csv(os.path.join(data1_dir, "dcgmi_metrics.csv"))
    df2 = pd.read_csv(os.path.join(data2_dir, "dcgmi_metrics.csv"))

    label1 = label_dir(data1_dir)
    label2 = label_dir(data2_dir)

    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16

    fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 8), sharex=True)

    window_size = 40

    for idx, metric in enumerate(metrics):
        if metric not in df1.columns or metric not in df2.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping.")
            continue

        scale = scale_dict.get(metric, 1)

        # d1 = df1[df1['gpu_id'] == gpu_id][metric] / scale
        # d2 = df2[df2['gpu_id'] == gpu_id][metric] / scale

        # ========== ==== reporting the averages for GPU utilization

        # Per-GPU data
        d1_raw = df1[df1['gpu_id'] == gpu_id][metric]
        d2_raw = df2[df2['gpu_id'] == gpu_id][metric]

        avg1 = d1_raw.mean() / scale
        avg2 = d2_raw.mean() / scale

        # Global (across all GPUs) mean
        gpu_ids_all = [0, 1, 2, 4]
        avg1_all = df1[df1['gpu_id'].isin(gpu_ids_all)].groupby('gpu_id')[metric].mean().mean() / scale
        avg2_all = df2[df2['gpu_id'].isin(gpu_ids_all)].groupby('gpu_id')[metric].mean().mean() / scale

        # Print results
        label_print = "GPUMem" if metric == "gpu_memory_usage" else metric.upper()
        print(f"[GPU{gpu_id}] {label_print:<7} - {label1}: {avg1:.2f} | {label2}: {avg2:.2f}")
        print(f"[ALL GPUs] {label_print:<7} - {label1}: {avg1_all:.2f} | {label2}: {avg2_all:.2f}")

        # Apply scaling and smoothing
        d1 = d1_raw / scale
        d2 = d2_raw / scale
        if smooth:
            d1 = d1.rolling(window=window_size, center=True).mean()
            d2 = d2.rolling(window=window_size, center=True).mean()

        # === === === = == = = == =====  end of the report


        if smooth:
            d1 = d1.rolling(window=window_size, center=True).mean()
            d2 = d2.rolling(window=window_size, center=True).mean()

        axs[idx].plot(d1, label=label1, linestyle='-', color='0.6', linewidth=1.5)
        axs[idx].plot(d2, label=label2, linestyle='--', color='0.0', linewidth=2)

        unit = ""
        if metric == "smact":
            unit = " (%)"
        elif metric == "power":
            unit = " (W)"
        elif metric == "gpu_memory_usage":
            unit = " (GB)"

        label = "GPU Memory" if metric == "gpu_memory_usage" else metric.upper()
        # axs[idx].set_ylabel(f"{label}{unit} - (GPU{gpu_id})")
        axs[idx].set_ylabel(f"{label}{unit}")

        if metric == "gpu_memory_usage":
            axs[idx].set_ylim(0, 40)

        axs[idx].legend(loc='upper right', frameon=False)
        axs[idx].grid(True, linestyle='--', linewidth=0.5, color='0.85')

    axs[-1].set_xlabel("Time (3-second intervals)")
    plt.tight_layout()
    filename = f"GPU{gpu_id}_all_metrics_comparison{'_smoothed' if smooth else ''}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot multiple metrics as subfigures for one GPU.")
    parser.add_argument("--dir1", type=str, required=True, help="First directory with dcgmi_metrics.csv")
    parser.add_argument("--dir2", type=str, required=True, help="Second directory with dcgmi_metrics.csv")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index (e.g., 0)")
    parser.add_argument("--smooth", action="store_true", help="Apply smoothing using rolling average")

    args = parser.parse_args()

    metrics = ["gpu_memory_usage", "smact", "power"]

    plot_subfigures_for_one_gpu(
        data1_dir=args.dir1,
        data2_dir=args.dir2,
        gpu_id=args.gpu,
        metrics=metrics,
        smooth=args.smooth
    )
