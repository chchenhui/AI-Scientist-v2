import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_7bc52d2190e34fa391c047ab21707aaa_proc_141122/experiment_data.npy",
        "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_18b75d33e2ce4c69883e353cb19086ef_proc_141123/experiment_data.npy",
        "experiments/2025-06-07_22-20-29_perturbation_ensemble_uq_attempt_0/logs/0-run/experiment_results/experiment_14b0e680daca45ff9f46f66729ada947_proc_141120/experiment_data.npy",
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
    for key in all_experiment_data[0]:
        dataset_info = all_experiment_data[0][key]
        if "losses" not in dataset_info or "metrics" not in dataset_info:
            continue

        train_losses_runs = [
            np.array(run[key]["losses"]["train"]) for run in all_experiment_data
        ]
        val_losses_runs = [
            np.array(run[key]["losses"]["val"]) for run in all_experiment_data
        ]
        train_auc_runs = [
            np.array(run[key]["metrics"]["train"]) for run in all_experiment_data
        ]
        val_auc_runs = [
            np.array(run[key]["metrics"]["val"]) for run in all_experiment_data
        ]

        train_losses_arr = np.vstack(train_losses_runs)
        val_losses_arr = np.vstack(val_losses_runs)
        train_auc_arr = np.vstack(train_auc_runs)
        val_auc_arr = np.vstack(val_auc_runs)

        n_runs, n_epochs = train_losses_arr.shape
        epochs = np.arange(1, n_epochs + 1)

        mean_train_losses = train_losses_arr.mean(axis=0)
        se_train_losses = train_losses_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        mean_val_losses = val_losses_arr.mean(axis=0)
        se_val_losses = val_losses_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        mean_train_auc = train_auc_arr.mean(axis=0)
        se_train_auc = train_auc_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        mean_val_auc = val_auc_arr.mean(axis=0)
        se_val_auc = val_auc_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)

        print(
            f"{key} dataset - Final Train Loss: {mean_train_losses[-1]:.4f} ± {se_train_losses[-1]:.4f}, Final Val Loss: {mean_val_losses[-1]:.4f} ± {se_val_losses[-1]:.4f}"
        )
        print(
            f"{key} dataset - Final Train AUC: {mean_train_auc[-1]:.4f} ± {se_train_auc[-1]:.4f}, Final Val AUC: {mean_val_auc[-1]:.4f} ± {se_val_auc[-1]:.4f}"
        )

        try:
            plt.figure()
            plt.errorbar(
                epochs,
                mean_train_losses,
                yerr=se_train_losses,
                label="Train Loss Mean ± SE",
            )
            plt.errorbar(
                epochs, mean_val_losses, yerr=se_val_losses, label="Val Loss Mean ± SE"
            )
            plt.title(f"Loss Curve\nMean ± SE across runs on {key} dataset")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{key}_loss_curve_agg.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss curve for {key}: {e}")
            plt.close()

        try:
            plt.figure()
            plt.errorbar(
                epochs, mean_train_auc, yerr=se_train_auc, label="Train AUC Mean ± SE"
            )
            plt.errorbar(
                epochs, mean_val_auc, yerr=se_val_auc, label="Val AUC Mean ± SE"
            )
            plt.title(f"AUC Curve\nMean ± SE across runs on {key} dataset")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{key}_auc_curve_agg.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated AUC curve for {key}: {e}")
            plt.close()
