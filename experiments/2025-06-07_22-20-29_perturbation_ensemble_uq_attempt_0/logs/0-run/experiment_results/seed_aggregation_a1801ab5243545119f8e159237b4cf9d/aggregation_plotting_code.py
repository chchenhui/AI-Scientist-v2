import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_ea3f7efe711046cb84aad3aa8201291c_proc_152738/experiment_data.npy",
        "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_cdb9f264bf8440bcb554ab2b6f318b9e_proc_152739/experiment_data.npy",
        "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_608b01f5455e4d448a0fd375847f92f7_proc_152740/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")
else:
    # Aggregate and plot per‐dataset
    for dataset in all_experiment_data[0].keys():
        # collect per‐run arrays
        train_runs, val_runs, auc_runs, des_runs = [], [], [], []
        epochs = None
        for exp in all_experiment_data:
            train_info = exp["losses"]["train"]
            val_info = exp["losses"]["val"]
            det_info = exp["metrics"]["detection"]
            if epochs is None:
                epochs = [d["epoch"] for d in train_info]
            train_runs.append([d["loss"] for d in train_info])
            val_runs.append([d["loss"] for d in val_info])
            auc_runs.append([d["auc"] for d in det_info])
            des_runs.append([d.get("DES", np.nan) for d in det_info])
        train_arr = np.array(train_runs)
        val_arr = np.array(val_runs)
        auc_arr = np.array(auc_runs)
        des_arr = np.array(des_runs)
        train_mean = train_arr.mean(axis=0)
        train_sem = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        val_mean = val_arr.mean(axis=0)
        val_sem = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        auc_mean = auc_arr.mean(axis=0)
        auc_sem = auc_arr.std(axis=0, ddof=1) / np.sqrt(auc_arr.shape[0])
        des_mean = des_arr.mean(axis=0)
        des_sem = des_arr.std(axis=0, ddof=1) / np.sqrt(des_arr.shape[0])

        try:
            plt.figure()
            plt.errorbar(
                epochs,
                train_mean,
                yerr=train_sem,
                label="Train Loss Mean ± SEM",
                capsize=3,
            )
            plt.errorbar(
                epochs, val_mean, yerr=val_sem, label="Val Loss Mean ± SEM", capsize=3
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} Loss Curve Mean ± SEM (Train vs Validation)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curve_mean_sem.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {dataset}: {e}")
            plt.close()

        try:
            plt.figure()
            plt.errorbar(
                epochs,
                auc_mean,
                yerr=auc_sem,
                marker="o",
                label="Detection AUC Mean ± SEM",
                capsize=3,
            )
            plt.xlabel("Epoch")
            plt.ylabel("Detection AUC")
            plt.title(f"{dataset} Detection AUC Mean ± SEM")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_detection_auc_mean_sem.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated detection AUC plot for {dataset}: {e}")
            plt.close()

        try:
            plt.figure()
            plt.errorbar(
                epochs,
                des_mean,
                yerr=des_sem,
                marker="o",
                label="DES Mean ± SEM",
                capsize=3,
            )
            plt.xlabel("Epoch")
            plt.ylabel("DES")
            plt.title(f"{dataset} DES Mean ± SEM")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_des_mean_sem.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated DES plot for {dataset}: {e}")
            plt.close()

        # Print final‐epoch summary
        print(
            f"{dataset} Final Metrics: "
            f"Train Loss {train_mean[-1]:.4f}±{train_sem[-1]:.4f}, "
            f"Val Loss {val_mean[-1]:.4f}±{val_sem[-1]:.4f}, "
            f"Detection AUC {auc_mean[-1]:.4f}±{auc_sem[-1]:.4f}, "
            f"DES {des_mean[-1]:.4f}±{des_sem[-1]:.4f}"
        )
