import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load all experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_ef7f284c309a41b18eb6941e48ff0218_proc_3949278/experiment_data.npy",
        "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_1e189401527a4907a1c4e2288e4f1b00_proc_3949279/experiment_data.npy",
        "None/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        try:
            exp = np.load(
                os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
                allow_pickle=True,
            ).item()
            all_experiment_data.append(exp)
        except Exception as e:
            print(f"Error loading {experiment_data_path}: {e}")
    # Determine unique learning rates
    lr_keys = set()
    for exp in all_experiment_data:
        lr_keys.update(exp.get("learning_rate", {}).keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []
    lr_keys = set()

# Print aggregated final validation metrics (mean ± SE)
for lr in sorted(lr_keys, key=lambda x: float(x)):
    final_losses, final_rets = [], []
    for exp in all_experiment_data:
        lr_data = exp["learning_rate"].get(lr, {})
        ds = lr_data.get("synthetic", {})
        if ds:
            final_losses.append(ds["losses"]["val"][-1])
            final_rets.append(ds["metrics"]["val"][-1])
    if final_losses:
        vals = np.array(final_losses)
        rets = np.array(final_rets)
        mean_loss = vals.mean()
        sem_loss = vals.std(ddof=1) / np.sqrt(len(vals))
        mean_ret = rets.mean()
        sem_ret = rets.std(ddof=1) / np.sqrt(len(rets))
        print(
            f"LR {lr} Mean Final Val Loss: {mean_loss:.4f} ± {sem_loss:.4f}, Mean Final Val Retention: {mean_ret:.4f} ± {sem_ret:.4f}"
        )

# Plot mean ± SE loss curves
try:
    plt.figure()
    for lr in sorted(lr_keys, key=lambda x: float(x)):
        train_runs, val_runs = [], []
        for exp in all_experiment_data:
            lr_data = exp["learning_rate"].get(lr, {})
            ds = lr_data.get("synthetic", {})
            if ds:
                train_runs.append(ds["losses"]["train"])
                val_runs.append(ds["losses"]["val"])
        if not train_runs:
            continue
        min_len = min(len(r) for r in train_runs)
        train_arr = np.array([r[:min_len] for r in train_runs])
        val_arr = np.array([r[:min_len] for r in val_runs])
        epochs = np.arange(min_len)
        mean_train = train_arr.mean(axis=0)
        sem_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        mean_val = val_arr.mean(axis=0)
        sem_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        plt.errorbar(epochs, mean_train, yerr=sem_train, label=f"train lr={lr}")
        plt.errorbar(
            epochs, mean_val, yerr=sem_val, linestyle="--", label=f"val lr={lr}"
        )
    plt.title("Mean ± SE Loss Curves Across Learning Rates\nDataset: synthetic")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_mean_se_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot mean ± SE memory retention ratios
try:
    plt.figure()
    for lr in sorted(lr_keys, key=lambda x: float(x)):
        train_runs, val_runs = [], []
        for exp in all_experiment_data:
            lr_data = exp["learning_rate"].get(lr, {})
            ds = lr_data.get("synthetic", {})
            if ds:
                train_runs.append(ds["metrics"]["train"])
                val_runs.append(ds["metrics"]["val"])
        if not train_runs:
            continue
        min_len = min(len(r) for r in train_runs)
        train_arr = np.array([r[:min_len] for r in train_runs])
        val_arr = np.array([r[:min_len] for r in val_runs])
        epochs = np.arange(min_len)
        mean_train = train_arr.mean(axis=0)
        sem_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        mean_val = val_arr.mean(axis=0)
        sem_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        plt.errorbar(epochs, mean_train, yerr=sem_train, label=f"train lr={lr}")
        plt.errorbar(
            epochs, mean_val, yerr=sem_val, linestyle="--", label=f"val lr={lr}"
        )
    plt.title(
        "Mean ± SE Memory Retention Ratios Across Learning Rates\nDataset: synthetic"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_mean_se_retention_ratios.png"))
    plt.close()
except Exception as e:
    print(f"Error creating retention ratio plot: {e}")
    plt.close()
