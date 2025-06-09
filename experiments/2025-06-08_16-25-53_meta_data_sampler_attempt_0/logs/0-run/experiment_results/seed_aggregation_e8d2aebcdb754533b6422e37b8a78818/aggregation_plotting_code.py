import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Paths to the three experiment_data.npy files
experiment_data_path_list = [
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_4a73eaea65a04a7095f224a36e30f169_proc_247309/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_81a8a3a45d6b49a3ac2d29fa4e5e3199_proc_247308/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_e759358712174d5e983304d5fc8a3b03_proc_247310/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Determine dataset names from the first run
dataset_names = list(all_experiment_data[0].keys()) if all_experiment_data else []

# Metrics to aggregate and plot (will skip any missing ones)
metrics_to_plot = {
    "train_loss": "Training Loss",
    "val_loss": "Validation Loss",
    "train_acc": "Training Accuracy",
    "val_acc": "Validation Accuracy",
    "corrs": "Spearman Correlation",
    "N_meta_history": "Meta-batch Size (N_meta)",
}

for metric_key, metric_name in metrics_to_plot.items():
    try:
        plt.figure()
        for ds in dataset_names:
            # Gather arrays for this metric across experiments
            arrs = []
            for exp_data in all_experiment_data:
                data = exp_data.get(ds, {})
                vals = data.get(metric_key, None)
                if vals is not None and len(vals) > 0:
                    arrs.append(np.array(vals))
            if not arrs:
                continue  # no data for this metric/dataset
            # Align lengths by truncation
            min_len = min(arr.shape[0] for arr in arrs)
            stacked = np.vstack([arr[:min_len] for arr in arrs])
            mean = np.mean(stacked, axis=0)
            sem = np.std(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            x = np.arange(1, min_len + 1)
            plt.plot(x, mean, marker="o", label=ds)
            plt.fill_between(x, mean - sem, mean + sem, alpha=0.2)
        plt.suptitle(f"{metric_name} Across Datasets")
        plt.title("Mean ± SEM across experiments")
        xlabel = "Epoch" if metric_key not in ["N_meta_history"] else "Meta-update Step"
        plt.xlabel(xlabel)
        plt.ylabel(metric_name)
        plt.legend()
        fname = f"{metric_key}_mean_sem.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {metric_key} mean sem plot: {e}")
        plt.close()
