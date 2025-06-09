import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load all experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_c5b491aafe1947f39e9c2308fe06a617_proc_385045/experiment_data.npy",
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_1b2ea6531bed4bfb8a350a13214f6289_proc_385046/experiment_data.npy",
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/logs/0-run/experiment_results/experiment_0175b29beba44f91b947647cb2b6688f_proc_385044/experiment_data.npy",
    ]
    all_experiment_data = []
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Prepare aggregated synthetic runs
synthetic_runs = []
for exp in all_experiment_data:
    try:
        synthetic_runs.append(exp["EPOCHS"]["synthetic"])
    except:
        continue

common_Es = (
    set.intersection(*[set(run.keys()) for run in synthetic_runs])
    if synthetic_runs
    else set()
)

# Plot mean ± SEM loss curves
try:
    plt.figure()
    for E in sorted(common_Es):
        train_curves = [run[E]["losses"]["train"] for run in synthetic_runs]
        val_curves = [run[E]["losses"]["val"] for run in synthetic_runs]
        train_arr = np.array(train_curves)
        val_arr = np.array(val_curves)
        epochs = np.arange(1, train_arr.shape[1] + 1)
        mean_train = train_arr.mean(axis=0)
        sem_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        mean_val = val_arr.mean(axis=0)
        sem_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        plt.errorbar(
            epochs, mean_train, yerr=sem_train, label=f"Train E={E}", capsize=3
        )
        plt.errorbar(
            epochs,
            mean_val,
            yerr=sem_val,
            linestyle="--",
            label=f"Val E={E}",
            capsize=3,
        )
    plt.suptitle("Synthetic Dataset")
    plt.title("Mean Train vs Validation Loss with SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_mean_sem_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss summary plot: {e}")
    plt.close()

# Plot mean ± SEM accuracy curves and print final metrics
try:
    plt.figure()
    for E in sorted(common_Es):
        train_acc = [run[E]["metrics"]["train"] for run in synthetic_runs]
        val_acc = [run[E]["metrics"]["val"] for run in synthetic_runs]
        train_arr = np.array(train_acc)
        val_arr = np.array(val_acc)
        epochs = np.arange(1, train_arr.shape[1] + 1)
        mean_train = train_arr.mean(axis=0)
        sem_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        mean_val = val_arr.mean(axis=0)
        sem_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        plt.errorbar(
            epochs, mean_train, yerr=sem_train, label=f"Train E={E}", capsize=3
        )
        plt.errorbar(
            epochs,
            mean_val,
            yerr=sem_val,
            linestyle="--",
            label=f"Val E={E}",
            capsize=3,
        )
        print(
            f"E={E} Final Accuracy -> Train: {mean_train[-1]:.4f} ± {sem_train[-1]:.4f}, Val: {mean_val[-1]:.4f} ± {sem_val[-1]:.4f}"
        )
    plt.suptitle("Synthetic Dataset")
    plt.title("Mean Train vs Validation Accuracy with SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_mean_sem_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy summary plot: {e}")
    plt.close()
