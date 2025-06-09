import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of relative experiment data paths
experiment_data_path_list = [
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/experiment_results/experiment_10fcb9c03a6349d0be26b639ce55b936_proc_17030/experiment_data.npy",
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/experiment_results/experiment_127c4c95906c46f9a8ba25bc6c34f578_proc_17031/experiment_data.npy",
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/experiment_results/experiment_40990ba0293d425393d648a3d85058c6_proc_17029/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Known learning rates
lrs = [1e-4, 1e-3, 1e-2]

# Collect all dataset names under 'multiple_synthetic'
dataset_names = set()
for d in all_experiment_data:
    dataset_names.update(d.get("multiple_synthetic", {}).keys())

# Aggregate and plot for each dataset
for ds_name in dataset_names:
    # Gather runs that contain this dataset
    train_metrics_list = []
    val_metrics_list = []
    train_losses_list = []
    val_losses_list = []
    for d in all_experiment_data:
        ds = d.get("multiple_synthetic", {}).get(ds_name)
        if ds is not None:
            train_metrics_list.append(ds["metrics"]["train"])
            val_metrics_list.append(ds["metrics"]["val"])
            train_losses_list.append(ds["losses"]["train"])
            val_losses_list.append(ds["losses"]["val"])
    if not train_metrics_list:
        continue

    # Stack and compute mean & standard error
    tm = np.stack(train_metrics_list, axis=0)
    vm = np.stack(val_metrics_list, axis=0)
    tl = np.stack(train_losses_list, axis=0)
    vl = np.stack(val_losses_list, axis=0)
    train_mean, train_sem = tm.mean(axis=0), tm.std(axis=0) / np.sqrt(len(tm))
    val_mean, val_sem = vm.mean(axis=0), vm.std(axis=0) / np.sqrt(len(vm))
    loss_train_mean, loss_train_sem = tl.mean(axis=0), tl.std(axis=0) / np.sqrt(len(tl))
    loss_val_mean, loss_val_sem = vl.mean(axis=0), vl.std(axis=0) / np.sqrt(len(vl))
    n_epochs = train_mean.shape[1]
    epochs = np.arange(n_epochs)

    # Aggregated accuracy with SEM
    try:
        plt.figure()
        for i, lr in enumerate(lrs):
            plt.errorbar(
                epochs,
                train_mean[i],
                yerr=train_sem[i],
                fmt="--",
                label=f"Train lr={lr}",
            )
            plt.errorbar(
                epochs, val_mean[i], yerr=val_sem[i], fmt="-", label=f"Val   lr={lr}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(
            f"{ds_name}: Weighted Accuracy (Aggregated)\nLeft: Train mean±SEM, Right: Val mean±SEM; synthetic dataset"
        )
        plt.legend()
        out_path = os.path.join(working_dir, f"{ds_name}_aggregated_accuracy_curve.png")
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds_name}: {e}")
        plt.close()

    # Aggregated loss with SEM
    try:
        plt.figure()
        for i, lr in enumerate(lrs):
            plt.errorbar(
                epochs,
                loss_train_mean[i],
                yerr=loss_train_sem[i],
                fmt="--",
                label=f"Train lr={lr}",
            )
            plt.errorbar(
                epochs,
                loss_val_mean[i],
                yerr=loss_val_sem[i],
                fmt="-",
                label=f"Val   lr={lr}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{ds_name}: Loss Curve (Aggregated)\nLeft: Train mean±SEM, Right: Val mean±SEM; synthetic dataset"
        )
        plt.legend()
        out_path = os.path.join(working_dir, f"{ds_name}_aggregated_loss_curve.png")
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()
