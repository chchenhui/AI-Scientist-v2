import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_fce85537044142eea1d3340b1d92973a_proc_234629/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_99ea040080fe490e8ba4afac99d6ae99_proc_234630/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_b78f5529dcd3453f9ebfa65101d0f2a0_proc_234628/experiment_data.npy",
]

# Load synthetic results from each run
all_synthetic = []
try:
    for path in experiment_data_path_list:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        syn = data["hyperparam_tuning_type_1"]["synthetic"]
        all_synthetic.append(syn)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract parameters and number of runs
param_values = all_synthetic[0]["param_values"]
n_runs = len(all_synthetic)

# Stack metrics across runs: shape (runs, params, epochs)
loss_train_runs = np.array([syn["losses"]["train"] for syn in all_synthetic])
loss_val_runs = np.array([syn["losses"]["val"] for syn in all_synthetic])
corrs_runs = np.array([syn["correlations"] for syn in all_synthetic])

# Compute mean and SEM
mean_train = np.mean(loss_train_runs, axis=0)
sem_train = np.std(loss_train_runs, axis=0) / np.sqrt(n_runs)
mean_val = np.mean(loss_val_runs, axis=0)
sem_val = np.std(loss_val_runs, axis=0) / np.sqrt(n_runs)
mean_corr = np.mean(corrs_runs, axis=0)
sem_corr = np.std(corrs_runs, axis=0) / np.sqrt(n_runs)

# Plot aggregated training/validation loss curves
try:
    plt.figure()
    for idx, p in enumerate(param_values):
        epochs = np.arange(1, mean_train.shape[1] + 1)
        plt.plot(epochs, mean_train[idx], label=f"{p} epochs train")
        plt.fill_between(
            epochs,
            mean_train[idx] - sem_train[idx],
            mean_train[idx] + sem_train[idx],
            alpha=0.2,
        )
        plt.plot(epochs, mean_val[idx], linestyle="--", label=f"{p} epochs val")
        plt.fill_between(
            epochs,
            mean_val[idx] - sem_val[idx],
            mean_val[idx] + sem_val[idx],
            alpha=0.2,
        )
    plt.suptitle("Synthetic Dataset Training/Validation Loss (Mean ± SE)")
    plt.title("Solid: Mean Train Loss, Dashed: Mean Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# Plot aggregated Spearman correlation curves
try:
    plt.figure()
    for idx, p in enumerate(param_values):
        epochs = np.arange(1, mean_corr.shape[1] + 1)
        plt.plot(epochs, mean_corr[idx], marker="o", label=f"{p} epochs")
        plt.fill_between(
            epochs,
            mean_corr[idx] - sem_corr[idx],
            mean_corr[idx] + sem_corr[idx],
            alpha=0.2,
        )
    plt.suptitle("Synthetic Dataset Spearman Correlation (Mean ± SE)")
    plt.title("Correlation of DVN Predictions vs True Contributions")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Corr")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_corr_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated Spearman correlation plot: {e}")
    plt.close()

# Print final epoch evaluation metrics
try:
    for idx, p in enumerate(param_values):
        final_mean = mean_corr[idx, -1]
        final_se = sem_corr[idx, -1]
        print(f"Param {p}: Final Spearman Corr = {final_mean:.3f} ± {final_se:.3f}")
except Exception as e:
    print(f"Error printing evaluation metrics: {e}")
