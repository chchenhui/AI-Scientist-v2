import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

exp_paths = [
    "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_d3388ee27ada46aca832d65717ab8d9d_proc_4061759/experiment_data.npy",
    "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_afaf04f731bd4975b5f1805f8133bb3f_proc_4061757/experiment_data.npy",
    "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_1bfb9a916ea04d0d8f4f43020b6e9adb_proc_4061758/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in exp_paths:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", os.getcwd()), p),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

lr_data = {}
for data in all_experiment_data:
    sd = data.get("learning_rate", {}).get("synthetic", {})
    lrs = sd.get("lrs", [])
    tr_losses = sd.get("losses", {}).get("train", [])
    vl_losses = sd.get("losses", {}).get("val", [])
    tr_aligns = sd.get("metrics", {}).get("train", [])
    vl_aligns = sd.get("metrics", {}).get("val", [])
    for lr, tr, vl, ta, va in zip(lrs, tr_losses, vl_losses, tr_aligns, vl_aligns):
        lr = float(lr)
        lr_data.setdefault(
            lr,
            {"train_losses": [], "val_losses": [], "train_align": [], "val_align": []},
        )
        lr_data[lr]["train_losses"].append(np.array(tr))
        lr_data[lr]["val_losses"].append(np.array(vl))
        lr_data[lr]["train_align"].append(np.array(ta))
        lr_data[lr]["val_align"].append(np.array(va))

first_lr = next(iter(lr_data)) if lr_data else None

# Plot 1: Aggregated Loss Curves
try:
    if first_lr is None:
        raise ValueError("No data available for aggregation")
    n_epochs = lr_data[first_lr]["train_losses"][0].shape[0]
    epochs = np.arange(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lr, stats in lr_data.items():
        arr_tr = np.stack(stats["train_losses"], axis=0)
        mean_tr = arr_tr.mean(axis=0)
        sem_tr = arr_tr.std(axis=0, ddof=1) / np.sqrt(arr_tr.shape[0])
        axes[0].plot(epochs, mean_tr, label=f"lr={lr} Mean")
        axes[0].fill_between(
            epochs, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.3, label=f"lr={lr} SEM"
        )
        arr_vl = np.stack(stats["val_losses"], axis=0)
        mean_vl = arr_vl.mean(axis=0)
        sem_vl = arr_vl.std(axis=0, ddof=1) / np.sqrt(arr_vl.shape[0])
        axes[1].plot(epochs, mean_vl, label=f"lr={lr} Mean")
        axes[1].fill_between(
            epochs, mean_vl - sem_vl, mean_vl + sem_vl, alpha=0.3, label=f"lr={lr} SEM"
        )
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Loss Curves\nLeft: Training Loss, Right: Validation Loss"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# Plot 2: Aggregated Alignment Curves
try:
    if first_lr is None:
        raise ValueError("No data available for aggregation")
    n_epochs = lr_data[first_lr]["train_align"][0].shape[0]
    epochs = np.arange(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lr, stats in lr_data.items():
        arr_tr = np.stack(stats["train_align"], axis=0)
        mean_tr = arr_tr.mean(axis=0)
        sem_tr = arr_tr.std(axis=0, ddof=1) / np.sqrt(arr_tr.shape[0])
        axes[0].plot(epochs, mean_tr, label=f"lr={lr} Mean")
        axes[0].fill_between(
            epochs, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.3, label=f"lr={lr} SEM"
        )
        arr_vl = np.stack(stats["val_align"], axis=0)
        mean_vl = arr_vl.mean(axis=0)
        sem_vl = arr_vl.std(axis=0, ddof=1) / np.sqrt(arr_vl.shape[0])
        axes[1].plot(epochs, mean_vl, label=f"lr={lr} Mean")
        axes[1].fill_between(
            epochs, mean_vl - sem_vl, mean_vl + sem_vl, alpha=0.3, label=f"lr={lr} SEM"
        )
    axes[0].set_title("Training Alignment")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Alignment (1-JSD)")
    axes[0].legend()
    axes[1].set_title("Validation Alignment")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Alignment (1-JSD)")
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Alignment Curves\nLeft: Training Alignment, Right: Validation Alignment"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated alignment plot: {e}")
    plt.close()
