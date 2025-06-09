import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load all experiment_data files
all_experiment_data = []
try:
    for fname in os.listdir(working_dir):
        if fname.startswith("experiment_data") and fname.endswith(".npy"):
            path = os.path.join(working_dir, fname)
            data = np.load(path, allow_pickle=True).item()
            all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate synthetic dataset results
loss_train_runs, loss_val_runs = [], []
metric_train_runs, metric_val_runs = [], []
params = None
for exp_data in all_experiment_data:
    d = exp_data.get("learning_rate", {}).get("synthetic", {})
    if not d:
        continue
    if params is None:
        params = d.get("params", [])
    loss_train_runs.append(np.array(d.get("losses", {}).get("train", [])))
    loss_val_runs.append(np.array(d.get("losses", {}).get("val", [])))
    metric_train_runs.append(np.array(d.get("metrics", {}).get("train", [])))
    metric_val_runs.append(np.array(d.get("metrics", {}).get("val", [])))

if loss_train_runs:
    loss_train_arr = np.stack(loss_train_runs, axis=0)
    loss_val_arr = np.stack(loss_val_runs, axis=0)
    metric_train_arr = np.stack(metric_train_runs, axis=0)
    metric_val_arr = np.stack(metric_val_runs, axis=0)
    runs = loss_train_arr.shape[0]
    # Compute mean and SEM
    loss_train_mean = loss_train_arr.mean(axis=0)
    loss_train_sem = loss_train_arr.std(axis=0, ddof=1) / np.sqrt(runs)
    loss_val_mean = loss_val_arr.mean(axis=0)
    loss_val_sem = loss_val_arr.std(axis=0, ddof=1) / np.sqrt(runs)
    metric_train_mean = metric_train_arr.mean(axis=0)
    metric_train_sem = metric_train_arr.std(axis=0, ddof=1) / np.sqrt(runs)
    metric_val_mean = metric_val_arr.mean(axis=0)
    metric_val_sem = metric_val_arr.std(axis=0, ddof=1) / np.sqrt(runs)
    epochs = np.arange(1, loss_train_mean.shape[1] + 1)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, lr in enumerate(params):
            axes[0].plot(epochs, loss_train_mean[i], label=f"{lr}")
            axes[0].fill_between(
                epochs,
                loss_train_mean[i] - loss_train_sem[i],
                loss_train_mean[i] + loss_train_sem[i],
                alpha=0.3,
            )
            axes[1].plot(epochs, loss_val_mean[i], label=f"{lr}")
            axes[1].fill_between(
                epochs,
                loss_val_mean[i] - loss_val_sem[i],
                loss_val_mean[i] + loss_val_sem[i],
                alpha=0.3,
            )
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[0].legend(title="LR")
        axes[1].legend(title="LR")
        fig.suptitle(
            "Synthetic dataset Loss Curves (Aggregated)\nLeft: Training Loss, Right: Validation Loss"
        )
        fig.savefig(os.path.join(working_dir, "synthetic_loss_curves_aggregated.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, lr in enumerate(params):
            axes[0].plot(epochs, metric_train_mean[i], label=f"{lr}")
            axes[0].fill_between(
                epochs,
                metric_train_mean[i] - metric_train_sem[i],
                metric_train_mean[i] + metric_train_sem[i],
                alpha=0.3,
            )
            axes[1].plot(epochs, metric_val_mean[i], label=f"{lr}")
            axes[1].fill_between(
                epochs,
                metric_val_mean[i] - metric_val_sem[i],
                metric_val_mean[i] + metric_val_sem[i],
                alpha=0.3,
            )
        axes[0].set_title("Training AICR")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("AICR")
        axes[1].set_title("Validation AICR")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AICR")
        axes[0].legend(title="LR")
        axes[1].legend(title="LR")
        fig.suptitle(
            "Synthetic dataset AICR Curves (Aggregated)\nLeft: Training AICR, Right: Validation AICR"
        )
        fig.savefig(os.path.join(working_dir, "synthetic_AICR_curves_aggregated.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating aggregated AICR plot: {e}")
        plt.close()
