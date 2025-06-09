import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_55fe680d4e1e4631a505c306f14e5335_proc_106394/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_4d6ca5d96db34b7e93963d8603b43adb_proc_106393/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_3b585f5093ae4f82b3fd7e7838e8d2e9_proc_106395/experiment_data.npy",
]

# Load all runs
all_runs = []
try:
    for rel_path in experiment_data_path_list:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path), allow_pickle=True
        ).item()
        all_runs.append(data["adam_beta1"]["synthetic"])
except Exception as e:
    print(f"Error loading experiment data: {e}")

if all_runs:
    beta1_list = [0.5, 0.7, 0.9, 0.99]
    n_runs = len(all_runs)
    epochs = len(all_runs[0]["metrics"]["train"][0])
    xs = np.arange(1, epochs + 1)

    # Plotting helper
    def plot_mean_sem(metric_key, ylabel, fname, title):
        try:
            plt.figure()
            for i, b1 in enumerate(beta1_list):
                # collect across runs
                arrs = np.array(
                    [
                        run["metrics" if metric_key in run["metrics"] else "losses"][
                            metric_key
                        ][i]
                        for run in all_runs
                    ]
                )
                mean = arrs.mean(axis=0)
                sem = arrs.std(axis=0, ddof=1) / np.sqrt(n_runs)
                plt.errorbar(xs, mean, yerr=sem, label=f"β1={b1}", capsize=3)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{title} - synthetic dataset")
            plt.legend(title="Mean ± SEM")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating {fname}: {e}")
            plt.close()

    plot_mean_sem(
        "train",
        "Relative Error",
        "synthetic_training_error_mean.png",
        "Training Error vs Epoch",
    )
    plot_mean_sem(
        "val",
        "Relative Error",
        "synthetic_validation_error_mean.png",
        "Validation Error vs Epoch",
    )
    plot_mean_sem(
        "train",
        "Reconstruction Loss",
        "synthetic_training_loss_mean.png",
        "Training Loss vs Epoch",
    )
    plot_mean_sem(
        "val",
        "MSE on Test",
        "synthetic_validation_loss_mean.png",
        "Validation Loss vs Epoch",
    )
