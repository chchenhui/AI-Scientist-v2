import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_148cd612fb144f9b86924ed59cedac2d_proc_113826/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_de2eb3e1b07942c68e91c748b3032e72_proc_113825/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_1a66bfcde0b54161a711b21588068016_proc_113824/experiment_data.npy",
]

all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

hp_group = "adam_beta1"
dataset = "synthetic"
metric_key = "metrics"
loss_key = "losses"
beta1_list = [0.5, 0.7, 0.9, 0.99]

# Determine epoch range
epochs = len(all_experiment_data[0][hp_group][dataset][metric_key]["train"][0])
xs = np.arange(1, epochs + 1)

# Compute and print final validation error stats
try:
    final_vals = []
    for i, b1 in enumerate(beta1_list):
        vals = np.array(
            [
                exp[hp_group][dataset][metric_key]["val"][i]
                for exp in all_experiment_data
            ]
        )
        mean_final = np.mean(vals[:, -1])
        se_final = np.std(vals[:, -1], ddof=0) / np.sqrt(vals.shape[0])
        final_vals.append((b1, mean_final, se_final))
    print("Final validation relative error (mean ± SE):")
    for b1, mean_final, se_final in final_vals:
        print(f"β1={b1}: {mean_final:.4f} ± {se_final:.4f}")
except Exception as e:
    print(f"Error computing final validation metrics: {e}")

# Plot aggregated training error
try:
    plt.figure()
    for i, b1 in enumerate(beta1_list):
        arr = np.array(
            [
                exp[hp_group][dataset][metric_key]["train"][i]
                for exp in all_experiment_data
            ]
        )
        m = np.mean(arr, axis=0)
        se = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        plt.plot(xs, m, label=f"β1={b1} Mean")
        plt.fill_between(xs, m - se, m + se, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.title("Training Error vs Epoch - synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_training_error_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training error plot: {e}")
    plt.close()

# Plot aggregated validation error
try:
    plt.figure()
    for i, b1 in enumerate(beta1_list):
        arr = np.array(
            [
                exp[hp_group][dataset][metric_key]["val"][i]
                for exp in all_experiment_data
            ]
        )
        m = np.mean(arr, axis=0)
        se = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        plt.plot(xs, m, label=f"β1={b1} Mean")
        plt.fill_between(xs, m - se, m + se, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.title("Validation Error vs Epoch - synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_validation_error_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation error plot: {e}")
    plt.close()

# Plot aggregated training loss
try:
    plt.figure()
    for i, b1 in enumerate(beta1_list):
        arr = np.array(
            [
                exp[hp_group][dataset][loss_key]["train"][i]
                for exp in all_experiment_data
            ]
        )
        m = np.mean(arr, axis=0)
        se = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        plt.plot(xs, m, label=f"β1={b1} Mean")
        plt.fill_between(xs, m - se, m + se, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Training Loss vs Epoch - synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_training_loss_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot aggregated validation loss
try:
    plt.figure()
    for i, b1 in enumerate(beta1_list):
        arr = np.array(
            [exp[hp_group][dataset][loss_key]["val"][i] for exp in all_experiment_data]
        )
        m = np.mean(arr, axis=0)
        se = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        plt.plot(xs, m, label=f"β1={b1} Mean")
        plt.fill_between(xs, m - se, m + se, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE on Test")
    plt.title("Validation Loss vs Epoch - synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_validation_loss_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
