import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    os.path.join(
        os.getenv("AI_SCIENTIST_ROOT"),
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/"
        "logs/0-run/experiment_results/experiment_43c5b1e3bd914f33839294535e89daa4_proc_400995/experiment_data.npy",
    ),
    os.path.join(
        os.getenv("AI_SCIENTIST_ROOT"),
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/"
        "logs/0-run/experiment_results/experiment_b1aaa62174ec4019bdaead133de21454_proc_400996/experiment_data.npy",
    ),
    os.path.join(
        os.getenv("AI_SCIENTIST_ROOT"),
        "experiments/2025-06-09_06-11-57_tracecode_dynamic_contrastive_attempt_0/"
        "logs/0-run/experiment_results/experiment_970f702b3a6a4461a61665d7f19beca7_proc_400997/experiment_data.npy",
    ),
]

# Load all experiment data
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        exp = np.load(path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Aggregate synthetic projection_head_ablation results
agg = {"loss": {}, "acc": {}}
for exp in all_experiment_data:
    syn = exp.get("projection_head_ablation", {}).get("synthetic", {})
    for head, runs in syn.items():
        for epochs, d in runs.items():
            # losses
            agg["loss"].setdefault(head, {}).setdefault(
                epochs, {"train": [], "val": []}
            )
            agg["loss"][head][epochs]["train"].append(np.array(d["losses"]["train"]))
            agg["loss"][head][epochs]["val"].append(np.array(d["losses"]["val"]))
            # accuracy
            agg["acc"].setdefault(head, {}).setdefault(epochs, {"train": [], "val": []})
            agg["acc"][head][epochs]["train"].append(np.array(d["metrics"]["train"]))
            agg["acc"][head][epochs]["val"].append(np.array(d["metrics"]["val"]))

# Plot mean ± SEM for Loss
try:
    plt.figure()
    for head, runs in agg["loss"].items():
        for epochs, data in runs.items():
            arr_tr = np.stack(data["train"], axis=0)
            arr_va = np.stack(data["val"], axis=0)
            x = np.arange(1, arr_tr.shape[1] + 1)
            mean_tr = arr_tr.mean(axis=0)
            sem_tr = arr_tr.std(axis=0, ddof=1) / np.sqrt(arr_tr.shape[0])
            mean_va = arr_va.mean(axis=0)
            sem_va = arr_va.std(axis=0, ddof=1) / np.sqrt(arr_va.shape[0])
            plt.plot(x, mean_tr, label=f"{head}_{epochs}_train_mean")
            plt.fill_between(x, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
            plt.plot(x, mean_va, label=f"{head}_{epochs}_val_mean")
            plt.fill_between(x, mean_va - sem_va, mean_va + sem_va, alpha=0.2)
    plt.title("Synthetic dataset: Mean Loss Curves (Train vs Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(working_dir, "synthetic_loss_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss mean-SEM plot: {e}")
    plt.close()

# Plot mean ± SEM for Accuracy
try:
    plt.figure()
    for head, runs in agg["acc"].items():
        for epochs, data in runs.items():
            arr_tr = np.stack(data["train"], axis=0)
            arr_va = np.stack(data["val"], axis=0)
            x = np.arange(1, arr_tr.shape[1] + 1)
            mean_tr = arr_tr.mean(axis=0)
            sem_tr = arr_tr.std(axis=0, ddof=1) / np.sqrt(arr_tr.shape[0])
            mean_va = arr_va.mean(axis=0)
            sem_va = arr_va.std(axis=0, ddof=1) / np.sqrt(arr_va.shape[0])
            plt.plot(x, mean_tr, label=f"{head}_{epochs}_train_acc_mean")
            plt.fill_between(x, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
            plt.plot(x, mean_va, label=f"{head}_{epochs}_val_acc_mean")
            plt.fill_between(x, mean_va - sem_va, mean_va + sem_va, alpha=0.2)
    plt.title("Synthetic dataset: Mean Retrieval Accuracy Curves (Train vs Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy mean-SEM plot: {e}")
    plt.close()
