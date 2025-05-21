import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load and aggregate data across runs
try:
    experiment_data_path_list = [
        "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_88ad674defc54627a13ed4a86fb95390_proc_4007053/experiment_data.npy",
        "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_7eeac2a50f9146109aca34e1a7661420_proc_4007054/experiment_data.npy",
        "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_922c3b95d2b7449a9c9eb92fb78f420b_proc_4007055/experiment_data.npy",
    ]
    all_experiment_data = []
    for rel_path in experiment_data_path_list:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    dataset = "synthetic"
    # Assume all runs share the same lrs
    sd0 = all_experiment_data[0].get("learning_rate", {}).get(dataset, {})
    lrs = sd0.get("lrs", [])
    n_runs = len(all_experiment_data)
    # Collect per-run metrics
    train_losses_list = []
    val_losses_list = []
    train_align_list = []
    val_align_list = []
    for d in all_experiment_data:
        sd = d.get("learning_rate", {}).get(dataset, {})
        train_losses_list.append(sd.get("losses", {}).get("train", []))
        val_losses_list.append(sd.get("losses", {}).get("val", []))
        train_align_list.append(sd.get("metrics", {}).get("train", []))
        val_align_list.append(sd.get("metrics", {}).get("val", []))
    # Convert to arrays (shape: n_runs x n_lrs x n_epochs)
    train_losses_arr = np.array(train_losses_list, dtype=float)
    val_losses_arr = np.array(val_losses_list, dtype=float)
    train_align_arr = np.array(train_align_list, dtype=float)
    val_align_arr = np.array(val_align_list, dtype=float)
    # Compute mean and standard error across runs
    mean_train_losses = np.mean(train_losses_arr, axis=0)
    se_train_losses = np.std(train_losses_arr, axis=0, ddof=1) / np.sqrt(n_runs)
    mean_val_losses = np.mean(val_losses_arr, axis=0)
    se_val_losses = np.std(val_losses_arr, axis=0, ddof=1) / np.sqrt(n_runs)
    mean_train_align = np.mean(train_align_arr, axis=0)
    se_train_align = np.std(train_align_arr, axis=0, ddof=1) / np.sqrt(n_runs)
    mean_val_align = np.mean(val_align_arr, axis=0)
    se_val_align = np.std(val_align_arr, axis=0, ddof=1) / np.sqrt(n_runs)
    # Print final validation metrics
    for idx, lr in enumerate(lrs):
        print(
            f"LR={lr} Final Val Loss: {mean_val_losses[idx,-1]:.4f} ± {se_val_losses[idx,-1]:.4f}"
        )
        print(
            f"LR={lr} Final Val Alignment: {mean_val_align[idx,-1]:.4f} ± {se_val_align[idx,-1]:.4f}"
        )
except Exception as e:
    print(f"Error loading or aggregating data: {e}")

# Plot 1: Loss curves with mean ± SE
try:
    if "mean_train_losses" not in locals():
        raise ValueError("Aggregated data unavailable for losses")
    epochs = np.arange(1, mean_train_losses.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, lr in enumerate(lrs):
        axes[0].plot(epochs, mean_train_losses[i], label=f"lr={lr}")
        axes[0].fill_between(
            epochs,
            mean_train_losses[i] - se_train_losses[i],
            mean_train_losses[i] + se_train_losses[i],
            alpha=0.2,
        )
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    for i, lr in enumerate(lrs):
        axes[1].plot(epochs, mean_val_losses[i], label=f"lr={lr}")
        axes[1].fill_between(
            epochs,
            mean_val_losses[i] - se_val_losses[i],
            mean_val_losses[i] + se_val_losses[i],
            alpha=0.2,
        )
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Loss Curves\nLeft: Training Loss, Right: Validation Loss"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: Alignment curves with mean ± SE
try:
    if "mean_train_align" not in locals():
        raise ValueError("Aggregated data unavailable for alignment")
    epochs = np.arange(1, mean_train_align.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, lr in enumerate(lrs):
        axes[0].plot(epochs, mean_train_align[i], label=f"lr={lr}")
        axes[0].fill_between(
            epochs,
            mean_train_align[i] - se_train_align[i],
            mean_train_align[i] + se_train_align[i],
            alpha=0.2,
        )
    axes[0].set_title("Training Alignment")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Alignment (1-JSD)")
    axes[0].legend()
    for i, lr in enumerate(lrs):
        axes[1].plot(epochs, mean_val_align[i], label=f"lr={lr}")
        axes[1].fill_between(
            epochs,
            mean_val_align[i] - se_val_align[i],
            mean_val_align[i] + se_val_align[i],
            alpha=0.2,
        )
    axes[1].set_title("Validation Alignment")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Alignment (1-JSD)")
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Alignment Curves\nLeft: Training Alignment, Right: Validation Alignment"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment curves plot: {e}")
    plt.close()
