import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Paths to each experiment_data.npy relative to AI_SCIENTIST_ROOT
experiment_data_path_list = [
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_5b805369d8964298a8b18633656a5e20_proc_2375579/experiment_data.npy",
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_a34a721fa6494b8c8ce4b52cdbcbb957_proc_2375580/experiment_data.npy",
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_7e5bc3d7000c49b49ca382bc9368bc9a_proc_2375581/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate per-dataset metrics
for dataset_name in all_experiment_data[0].keys():
    try:
        # Collect losses and CES curves
        loss_train_list = [
            exp[dataset_name]["losses"]["train"] for exp in all_experiment_data
        ]
        loss_val_list = [
            exp[dataset_name]["losses"]["val"] for exp in all_experiment_data
        ]
        ces_train_list = [
            exp[dataset_name]["metrics"]["train"] for exp in all_experiment_data
        ]
        ces_val_list = [
            exp[dataset_name]["metrics"]["val"] for exp in all_experiment_data
        ]
        # Truncate to shortest length
        min_loss_len = min(len(x) for x in loss_train_list)
        min_ces_len = min(len(x) for x in ces_train_list)
        loss_train_arr = np.vstack([x[:min_loss_len] for x in loss_train_list])
        loss_val_arr = np.vstack([x[:min_loss_len] for x in loss_val_list])
        ces_train_arr = np.vstack([x[:min_ces_len] for x in ces_train_list])
        ces_val_arr = np.vstack([x[:min_ces_len] for x in ces_val_list])
        # Compute mean and SEM
        loss_train_mean = loss_train_arr.mean(axis=0)
        loss_train_sem = loss_train_arr.std(axis=0, ddof=1) / np.sqrt(
            loss_train_arr.shape[0]
        )
        loss_val_mean = loss_val_arr.mean(axis=0)
        loss_val_sem = loss_val_arr.std(axis=0, ddof=1) / np.sqrt(loss_val_arr.shape[0])
        ces_train_mean = ces_train_arr.mean(axis=0)
        ces_train_sem = ces_train_arr.std(axis=0, ddof=1) / np.sqrt(
            ces_train_arr.shape[0]
        )
        ces_val_mean = ces_val_arr.mean(axis=0)
        ces_val_sem = ces_val_arr.std(axis=0, ddof=1) / np.sqrt(ces_val_arr.shape[0])
        epochs_loss = np.arange(1, min_loss_len + 1)
        epochs_ces = np.arange(1, min_ces_len + 1)
    except Exception as e:
        print(f"Error aggregating metrics for {dataset_name}: {e}")
        continue

    # Plot aggregated loss curves with SEM
    try:
        plt.figure()
        plt.errorbar(
            epochs_loss,
            loss_train_mean,
            yerr=loss_train_sem,
            label="Train Loss",
            capsize=3,
        )
        plt.errorbar(
            epochs_loss, loss_val_mean, yerr=loss_val_sem, label="Val Loss", capsize=3
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Aggregated Loss Curves with SEM - {dataset_name}")
        plt.legend()
        fname = f"loss_sem_{dataset_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dataset_name}: {e}")
        plt.close()

    # Plot aggregated CES curves with SEM
    try:
        plt.figure()
        plt.errorbar(
            epochs_ces, ces_train_mean, yerr=ces_train_sem, label="Train CES", capsize=3
        )
        plt.errorbar(
            epochs_ces, ces_val_mean, yerr=ces_val_sem, label="Val CES", capsize=3
        )
        plt.xlabel("Epoch")
        plt.ylabel("CES")
        plt.title(f"Aggregated CES Curves with SEM - {dataset_name}")
        plt.legend()
        fname = f"ces_sem_{dataset_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CES plot for {dataset_name}: {e}")
        plt.close()

    # Print final aggregated metrics
    try:
        print(
            f"{dataset_name} Final Train Loss Mean ± SEM: {loss_train_mean[-1]:.4f} ± {loss_train_sem[-1]:.4f}"
        )
        print(
            f"{dataset_name} Final Val   Loss Mean ± SEM: {loss_val_mean[-1]:.4f} ± {loss_val_sem[-1]:.4f}"
        )
        print(
            f"{dataset_name} Final Train CES  Mean ± SEM: {ces_train_mean[-1]:.4f} ± {ces_train_sem[-1]:.4f}"
        )
        print(
            f"{dataset_name} Final Val   CES  Mean ± SEM: {ces_val_mean[-1]:.4f} ± {ces_val_sem[-1]:.4f}"
        )
    except:
        pass
