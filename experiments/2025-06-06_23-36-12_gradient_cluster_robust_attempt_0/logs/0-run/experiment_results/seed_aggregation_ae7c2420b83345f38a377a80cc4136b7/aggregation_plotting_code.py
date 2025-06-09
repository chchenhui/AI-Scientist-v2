import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data from multiple runs
experiment_data_paths = [
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/experiment_results/experiment_d9f6b86a6c8248cbab4aa90ba817f260_proc_11504/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/experiment_results/experiment_cc3e08fa12a544a69ec69d976d9e1189_proc_11502/experiment_data.npy",
]
all_data = []
try:
    for path in experiment_data_paths:
        try:
            data = np.load(
                os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
            ).item()
            all_data.append(data)
        except Exception as e:
            print(f"Error loading experiment data from {path}: {e}")
except Exception as e:
    print(f"Error initializing data loading: {e}")

# Aggregate and plot per dataset
dataset_names = all_data[0].get("learning_rate", {}).keys() if all_data else []
for dataset in dataset_names:
    # collect runs for this dataset
    run_list = [
        exp["learning_rate"][dataset]
        for exp in all_data
        if dataset in exp.get("learning_rate", {})
    ]
    if not run_list:
        continue
    # stack metrics and losses: shape (runs, n_lrs, epochs)
    metrics_tr = np.stack([r["metrics"]["train"] for r in run_list], axis=0)
    metrics_val = np.stack([r["metrics"]["val"] for r in run_list], axis=0)
    loss_tr = np.stack([r["losses"]["train"] for r in run_list], axis=0)
    loss_val = np.stack([r["losses"]["val"] for r in run_list], axis=0)
    lrs = run_list[0]["lrs"]
    # compute mean and standard error
    m_tr_mean = metrics_tr.mean(axis=0)
    m_tr_sem = metrics_tr.std(axis=0) / np.sqrt(metrics_tr.shape[0])
    m_val_mean = metrics_val.mean(axis=0)
    m_val_sem = metrics_val.std(axis=0) / np.sqrt(metrics_val.shape[0])
    l_tr_mean = loss_tr.mean(axis=0)
    l_tr_sem = loss_tr.std(axis=0) / np.sqrt(loss_tr.shape[0])
    l_val_mean = loss_val.mean(axis=0)
    l_val_sem = loss_val.std(axis=0) / np.sqrt(loss_val.shape[0])
    # print final‐epoch metrics
    for i, lr in enumerate(lrs):
        print(
            f"{dataset} lr={lr} - Final Train WG Acc: {m_tr_mean[i,-1]:.4f} ± {m_tr_sem[i,-1]:.4f}, "
            f"Val WG Acc: {m_val_mean[i,-1]:.4f} ± {m_val_sem[i,-1]:.4f}"
        )
        print(
            f"{dataset} lr={lr} - Final Train Loss: {l_tr_mean[i,-1]:.4f} ± {l_tr_sem[i,-1]:.4f}, "
            f"Val Loss: {l_val_mean[i,-1]:.4f} ± {l_val_sem[i,-1]:.4f}"
        )
    # Plot aggregated worst‐group accuracy
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = np.arange(1, m_tr_mean.shape[1] + 1)
        for i, lr in enumerate(lrs):
            axes[0].errorbar(epochs, m_tr_mean[i], yerr=m_tr_sem[i], label=f"lr={lr}")
            axes[1].errorbar(epochs, m_val_mean[i], yerr=m_val_sem[i], label=f"lr={lr}")
        axes[0].set_title("Training WG Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Worst‐group Accuracy")
        axes[1].set_title("Validation WG Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Worst‐group Accuracy")
        axes[0].legend()
        axes[1].legend()
        fig.suptitle(
            f"{dataset.capitalize()} dataset - Worst‐group Accuracy (Aggregated)\nLeft: Training, Right: Validation"
        )
        plt.savefig(
            os.path.join(working_dir, f"{dataset}_worst_group_accuracy_agg.png")
        )
        plt.close(fig)
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dataset}: {e}")
        plt.close("all")
    # Plot aggregated loss curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = np.arange(1, l_tr_mean.shape[1] + 1)
        for i, lr in enumerate(lrs):
            axes[0].errorbar(epochs, l_tr_mean[i], yerr=l_tr_sem[i], label=f"lr={lr}")
            axes[1].errorbar(epochs, l_val_mean[i], yerr=l_val_sem[i], label=f"lr={lr}")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[0].legend()
        axes[1].legend()
        fig.suptitle(
            f"{dataset.capitalize()} dataset - Loss Curves (Aggregated)\nLeft: Training, Right: Validation"
        )
        plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curves_agg.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dataset}: {e}")
        plt.close("all")
