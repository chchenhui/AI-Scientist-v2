import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load all experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_70241602ee6f4d26a04db7350577e588_proc_382491/experiment_data.npy",
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_aa4c7b01b0f64f9d869be6c4211da64b_proc_382492/experiment_data.npy",
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_225a468a0a3c4355adc54655885e19e0_proc_382493/experiment_data.npy",
    ]
    all_experiment_data = []
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Determine datasets
dataset_names = all_experiment_data[0].keys() if all_experiment_data else []

# Aggregate and plot loss curves
for ds in dataset_names:
    try:
        # gather losses
        runs_train = [
            np.array(exp[ds]["losses"]["train"]) for exp in all_experiment_data
        ]
        runs_val = [np.array(exp[ds]["losses"]["val"]) for exp in all_experiment_data]
        min_epochs = min(arr.shape[0] for arr in runs_train + runs_val)
        runs_train = np.vstack([arr[:min_epochs] for arr in runs_train])
        runs_val = np.vstack([arr[:min_epochs] for arr in runs_val])
        epochs = np.arange(1, min_epochs + 1)
        mean_train = runs_train.mean(axis=0)
        sem_train = runs_train.std(axis=0, ddof=1) / np.sqrt(runs_train.shape[0])
        mean_val = runs_val.mean(axis=0)
        sem_val = runs_val.std(axis=0, ddof=1) / np.sqrt(runs_val.shape[0])
        plt.figure()
        plt.errorbar(
            epochs, mean_train, yerr=sem_train, label="Train Loss Mean ± SEM", capsize=3
        )
        plt.errorbar(
            epochs, mean_val, yerr=sem_val, label="Val Loss Mean ± SEM", capsize=3
        )
        plt.title(f"Loss Curves for {ds} dataset\nMean ± SEM over experiments")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss_curve_mean_sem.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

# Aggregate and plot accuracy curves
for ds in dataset_names:
    try:
        runs_train_acc = [
            np.array(exp[ds]["metrics"]["train"]) for exp in all_experiment_data
        ]
        runs_val_acc = [
            np.array(exp[ds]["metrics"]["val"]) for exp in all_experiment_data
        ]
        min_epochs_acc = min(arr.shape[0] for arr in runs_train_acc + runs_val_acc)
        runs_train_acc = np.vstack([arr[:min_epochs_acc] for arr in runs_train_acc])
        runs_val_acc = np.vstack([arr[:min_epochs_acc] for arr in runs_val_acc])
        epochs_acc = np.arange(1, min_epochs_acc + 1)
        mean_train_acc = runs_train_acc.mean(axis=0)
        sem_train_acc = runs_train_acc.std(axis=0, ddof=1) / np.sqrt(
            runs_train_acc.shape[0]
        )
        mean_val_acc = runs_val_acc.mean(axis=0)
        sem_val_acc = runs_val_acc.std(axis=0, ddof=1) / np.sqrt(runs_val_acc.shape[0])
        plt.figure()
        plt.errorbar(
            epochs_acc,
            mean_train_acc,
            yerr=sem_train_acc,
            label="Train Acc Mean ± SEM",
            capsize=3,
        )
        plt.errorbar(
            epochs_acc,
            mean_val_acc,
            yerr=sem_val_acc,
            label="Val Acc Mean ± SEM",
            capsize=3,
        )
        plt.title(f"Accuracy Curves for {ds} dataset\nMean ± SEM over experiments")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_accuracy_curve_mean_sem.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds}: {e}")
        plt.close()

# Print final evaluation metrics
for ds in dataset_names:
    try:
        final_vals = [exp[ds]["metrics"]["val"][-1] for exp in all_experiment_data]
        mean_final = np.mean(final_vals)
        sem_final = np.std(final_vals, ddof=1) / np.sqrt(len(final_vals))
        print(f"Dataset {ds}: Final Val Acc = {mean_final:.3f} ± {sem_final:.3f}")
    except Exception as e:
        print(f"Error computing final metric for {ds}: {e}")
