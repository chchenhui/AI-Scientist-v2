import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_cde2566319a94c8ab6b12f9bf4bf3893_proc_144144/experiment_data.npy",
    "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_de20a07534e64095bd1e496b85c4cd9a_proc_144145/experiment_data.npy",
    "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_f1bbd7a781534559822427943d2c515a_proc_144143/experiment_data.npy",
]

all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        exp = np.load(path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")

else:
    # Gather all dataset names
    datasets = set()
    for exp in all_experiment_data:
        datasets.update(exp.keys())
    for dataset in datasets:
        # collect runs
        train_loss_runs, val_loss_runs = [], []
        train_auc_runs, val_auc_runs = [], []
        epochs_loss, epochs_auc = None, None
        for exp in all_experiment_data:
            if dataset not in exp:
                continue
            data = exp[dataset]
            best = max(data["metrics"]["val"], key=lambda d: d["auc"])
            bs, lr = best["bs"], best["lr"]
            # extract and sort curves
            tl = sorted(
                (d["epoch"], d["loss"])
                for d in data["losses"]["train"]
                if d["bs"] == bs and d["lr"] == lr
            )
            vl = sorted(
                (d["epoch"], d["loss"])
                for d in data["losses"]["val"]
                if d["bs"] == bs and d["lr"] == lr
            )
            ta = sorted(
                (d["epoch"], d["auc"])
                for d in data["metrics"]["train"]
                if d["bs"] == bs and d["lr"] == lr
            )
            va = sorted(
                (d["epoch"], d["auc"])
                for d in data["metrics"]["val"]
                if d["bs"] == bs and d["lr"] == lr
            )
            el, sl = zip(*tl)
            _, vsl = zip(*vl)
            ea, sa = zip(*ta)
            _, vsa = zip(*va)
            if epochs_loss is None:
                epochs_loss = el
            if epochs_auc is None:
                epochs_auc = ea
            train_loss_runs.append(sl)
            val_loss_runs.append(vsl)
            train_auc_runs.append(sa)
            val_auc_runs.append(vsa)
        # aggregated loss plot
        if train_loss_runs:
            arr_tl = np.array(train_loss_runs)
            arr_vl = np.array(val_loss_runs)
            mean_tl = arr_tl.mean(axis=0)
            sem_tl = arr_tl.std(axis=0, ddof=1) / np.sqrt(arr_tl.shape[0])
            mean_vl = arr_vl.mean(axis=0)
            sem_vl = arr_vl.std(axis=0, ddof=1) / np.sqrt(arr_vl.shape[0])
            try:
                plt.figure()
                plt.errorbar(epochs_loss, mean_tl, yerr=sem_tl, label="Train Loss")
                plt.errorbar(epochs_loss, mean_vl, yerr=sem_vl, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(
                    f"{dataset} Loss Curve (Aggregated Mean ± SEM)\nLeft: Train, Right: Validation"
                )
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{dataset}_loss_aggregated.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating aggregated loss plot for {dataset}: {e}")
                plt.close()
        # aggregated AUC plot
        if train_auc_runs:
            arr_ta = np.array(train_auc_runs)
            arr_va = np.array(val_auc_runs)
            mean_ta = arr_ta.mean(axis=0)
            sem_ta = arr_ta.std(axis=0, ddof=1) / np.sqrt(arr_ta.shape[0])
            mean_va = arr_va.mean(axis=0)
            sem_va = arr_va.std(axis=0, ddof=1) / np.sqrt(arr_va.shape[0])
            try:
                plt.figure()
                plt.errorbar(epochs_auc, mean_ta, yerr=sem_ta, label="Train AUC")
                plt.errorbar(epochs_auc, mean_va, yerr=sem_va, label="Val AUC")
                plt.xlabel("Epoch")
                plt.ylabel("AUC")
                plt.title(
                    f"{dataset} AUC Curve (Aggregated Mean ± SEM)\nLeft: Train, Right: Validation"
                )
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{dataset}_auc_aggregated.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating aggregated AUC plot for {dataset}: {e}")
                plt.close()
