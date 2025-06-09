import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load multiple experiment_data files
try:
    experiment_data_path_list = [
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_614fc711dd7d41c192f0c78e7923b238_proc_395097/experiment_data.npy",
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_4d809ff3638143aca86191056c6d8a82_proc_395098/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Aggregate runs by E for synthetic dataset
runs_per_E = {}
for exp in all_experiment_data:
    synthetic = exp.get("EPOCHS", {}).get("synthetic", {})
    for E, run in synthetic.items():
        runs_per_E.setdefault(E, []).append(run)

# Plot aggregated loss curves with mean ± SEM
try:
    plt.figure()
    for E, run_list in runs_per_E.items():
        if not run_list:
            continue
        train_losses = [np.array(r["losses"]["train"]) for r in run_list]
        val_losses = [np.array(r["losses"]["val"]) for r in run_list]
        min_len = min(
            min(arr.shape[0] for arr in train_losses),
            min(arr.shape[0] for arr in val_losses),
        )
        train_arr = np.vstack([arr[:min_len] for arr in train_losses])
        val_arr = np.vstack([arr[:min_len] for arr in val_losses])
        epochs = np.arange(1, min_len + 1)
        mean_train = train_arr.mean(axis=0)
        sem_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        mean_val = val_arr.mean(axis=0)
        sem_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        plt.plot(epochs, mean_train, label=f"Train E={E}")
        plt.fill_between(
            epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.3
        )
        plt.plot(epochs, mean_val, linestyle="--", label=f"Val E={E}")
        plt.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.3)
    plt.suptitle("Synthetic Dataset")
    plt.title("Train vs Validation Loss Curves with Mean±SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_mean_sem_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# Plot aggregated retrieval accuracy with mean ± SEM
try:
    plt.figure()
    for E, run_list in runs_per_E.items():
        if not run_list:
            continue
        train_acc = [np.array(r["metrics"]["train"]) for r in run_list]
        val_acc = [np.array(r["metrics"]["val"]) for r in run_list]
        min_len = min(
            min(arr.shape[0] for arr in train_acc), min(arr.shape[0] for arr in val_acc)
        )
        train_arr = np.vstack([arr[:min_len] for arr in train_acc])
        val_arr = np.vstack([arr[:min_len] for arr in val_acc])
        epochs = np.arange(1, min_len + 1)
        mean_train = train_arr.mean(axis=0)
        sem_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        mean_val = val_arr.mean(axis=0)
        sem_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        plt.plot(epochs, mean_train, label=f"Train E={E}")
        plt.fill_between(
            epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.3
        )
        plt.plot(epochs, mean_val, linestyle="--", label=f"Val E={E}")
        plt.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.3)
    plt.suptitle("Synthetic Dataset")
    plt.title("Train vs Validation Retrieval Accuracy with Mean±SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_mean_sem_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()
