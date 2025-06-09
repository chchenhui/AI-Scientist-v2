import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# List of provided experiment data relative paths
experiment_data_paths = [
    "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_869bf7fad75b4c15bd002f01a8a6dfc0_proc_164440/experiment_data.npy",
    "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_21d62251fddf4f238c0268462b6bca8d_proc_164442/experiment_data.npy",
    "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_8c608bb7f9e8449fb88a79cbe43f43dc_proc_164441/experiment_data.npy",
]

# Load all experiment data into a list
try:
    all_experiment_data = []
    for rel_path in experiment_data_paths:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Identify all ablations
all_ablations = set()
for exp in all_experiment_data:
    all_ablations.update(exp.keys())

# Iterate ablations and datasets
for ablation in all_ablations:
    # collect dataset names under this ablation
    ds_names = set()
    for exp in all_experiment_data:
        if ablation in exp:
            ds_names.update(exp[ablation].keys())
    for ds_name in ds_names:
        # Aggregate training and validation losses
        train_runs, val_runs, epochs = [], [], None
        for exp in all_experiment_data:
            if ablation in exp and ds_name in exp[ablation]:
                ds_data = exp[ablation][ds_name]
                e = [d["epoch"] for d in ds_data["losses"]["train"]]
                if epochs is None:
                    epochs = e
                train_runs.append([d["loss"] for d in ds_data["losses"]["train"]])
                val_runs.append([d["loss"] for d in ds_data["losses"]["val"]])
        if not train_runs:
            continue
        train_arr = np.array(train_runs)
        val_arr = np.array(val_runs)
        mean_train = np.mean(train_arr, axis=0)
        se_train = np.std(train_arr, ddof=1, axis=0) / np.sqrt(train_arr.shape[0])
        mean_val = np.mean(val_arr, axis=0)
        se_val = np.std(val_arr, ddof=1, axis=0) / np.sqrt(val_arr.shape[0])
        try:
            plt.figure()
            plt.errorbar(
                epochs, mean_train, yerr=se_train, label="Train Loss", capsize=3
            )
            plt.errorbar(epochs, mean_val, yerr=se_val, label="Val Loss", capsize=3)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Loss Curves Aggregated ({ablation})")
            plt.legend()
            fname = f"{ds_name}_{ablation}_loss_curves_agg.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated {ds_name} {ablation} loss plot: {e}")
            plt.close()
        # Aggregate detection AUC metrics
        det_runs_vote, det_runs_kl, det_epochs = [], [], None
        for exp in all_experiment_data:
            if ablation in exp and ds_name in exp[ablation]:
                metrics = exp[ablation][ds_name]["metrics"]["detection"]
                det_epochs = [m["epoch"] for m in metrics]
                det_runs_vote.append([m["auc_vote"] for m in metrics])
                det_runs_kl.append([m["auc_kl"] for m in metrics])
        if not det_runs_vote:
            continue
        vote_arr = np.array(det_runs_vote)
        kl_arr = np.array(det_runs_kl)
        mean_vote = np.mean(vote_arr, axis=0)
        se_vote = np.std(vote_arr, ddof=1, axis=0) / np.sqrt(vote_arr.shape[0])
        mean_kl = np.mean(kl_arr, axis=0)
        se_kl = np.std(kl_arr, ddof=1, axis=0) / np.sqrt(kl_arr.shape[0])
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].errorbar(det_epochs, mean_vote, yerr=se_vote, marker="o", capsize=3)
            axes[0].set_title("Vote AUC")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("AUC")
            axes[1].errorbar(det_epochs, mean_kl, yerr=se_kl, marker="o", capsize=3)
            axes[1].set_title("KL AUC")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("AUC")
            plt.suptitle(
                f"{ds_name} Detection AUC Aggregated ({ablation})\nLeft: Vote AUC, Right: KL AUC"
            )
            fname = f"{ds_name}_{ablation}_detection_auc_agg.png"
            fig.savefig(os.path.join(working_dir, fname))
            plt.close(fig)
        except Exception as e:
            print(f"Error creating aggregated {ds_name} {ablation} detection plot: {e}")
            plt.close()
