import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load all experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_a5277dcb69a643d281445a6a6bcc1ba4_proc_3946599/experiment_data.npy",
        "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_96860331a4694cbd962fd21ef89ceee5_proc_3946601/experiment_data.npy",
        "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_b684f4a36dd74c988a170c3aab6505de_proc_3946600/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp.get("synthetic", {}))
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Aggregate losses and metrics
loss_trains = [e.get("losses", {}).get("train", []) for e in all_experiment_data]
loss_vals = [e.get("losses", {}).get("val", []) for e in all_experiment_data]
met_trains = [e.get("metrics", {}).get("train", []) for e in all_experiment_data]
met_vals = [e.get("metrics", {}).get("val", []) for e in all_experiment_data]


def compute_stats(list_of_lists):
    valid = [l for l in list_of_lists if len(l) > 0]
    if not valid:
        return None, None
    min_len = min(len(l) for l in valid)
    arr = np.array([l[:min_len] for l in valid])
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])
    return mean, se


lt_mean, lt_se = compute_stats(loss_trains)
lv_mean, lv_se = compute_stats(loss_vals)
mt_mean, mt_se = compute_stats(met_trains)
mv_mean, mv_se = compute_stats(met_vals)

# Print final means and SE
if lt_mean is not None:
    print(f"Final Train Loss Mean: {lt_mean[-1]:.4f} ± {lt_se[-1]:.4f}")
if lv_mean is not None:
    print(f"Final Val   Loss Mean: {lv_mean[-1]:.4f} ± {lv_se[-1]:.4f}")
if mt_mean is not None:
    print(f"Final Train Metric Mean: {mt_mean[-1]:.4f} ± {mt_se[-1]:.4f}")
if mv_mean is not None:
    print(f"Final Val   Metric Mean: {mv_mean[-1]:.4f} ± {mv_se[-1]:.4f}")

# Plot mean ± SE loss
try:
    if lt_mean is not None and lv_mean is not None:
        plt.figure()
        epochs = np.arange(1, len(lt_mean) + 1)
        plt.errorbar(epochs, lt_mean, yerr=lt_se, label="Train Loss Mean ± SE")
        plt.errorbar(epochs, lv_mean, yerr=lv_se, label="Val Loss Mean ± SE")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "Synthetic Dataset Loss Curves with Mean and SE\nTraining vs Validation Loss"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_mean_se.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss mean/se plot: {e}")
    plt.close()

# Plot mean ± SE metrics
try:
    if mt_mean is not None and mv_mean is not None:
        plt.figure()
        epochs = np.arange(1, len(mt_mean) + 1)
        plt.errorbar(epochs, mt_mean, yerr=mt_se, label="Train Metric Mean ± SE")
        plt.errorbar(epochs, mv_mean, yerr=mv_se, label="Val Metric Mean ± SE")
        plt.xlabel("Epoch")
        plt.ylabel("Memory Retention Ratio")
        plt.title(
            "Synthetic Dataset Metric Curves with Mean and SE\nMemory Retention over Epochs"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_metric_mean_se.png"))
        plt.close()
except Exception as e:
    print(f"Error creating metric mean/se plot: {e}")
    plt.close()

# Aggregate and plot generation sequences with mean ± SE
try:
    preds_list = [e.get("predictions", []) for e in all_experiment_data]
    gt_list = [e.get("ground_truth", []) for e in all_experiment_data]
    gt_mean, gt_se = compute_stats(gt_list)
    pd_mean, pd_se = compute_stats(preds_list)
    if gt_mean is not None and pd_mean is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        x_gt = np.arange(1, len(gt_mean) + 1)
        axs[0].errorbar(x_gt, gt_mean, yerr=gt_se, label="Mean ± SE")
        axs[0].set_title("Ground Truth Sequence (Mean ± SE)")
        axs[0].set_xlabel("Time Step")
        axs[0].set_ylabel("Token")
        axs[0].legend()
        x_pd = np.arange(1, len(pd_mean) + 1)
        axs[1].errorbar(x_pd, pd_mean, yerr=pd_se, color="orange", label="Mean ± SE")
        axs[1].set_title("Generated Samples (Mean ± SE)")
        axs[1].set_xlabel("Time Step")
        axs[1].legend()
        plt.suptitle(
            "Synthetic Dataset Generation Comparison\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.savefig(os.path.join(working_dir, "synthetic_generation_mean_se.png"))
        plt.close()
except Exception as e:
    print(f"Error creating generation mean/se plot: {e}")
    plt.close()
