import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    os.makedirs(working_dir, exist_ok=True)
except Exception:
    pass

# Paths to all experiment_data.npy files
experiment_data_path_list = [
    "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_b67436265d954a79bda0a9319d10bff0_proc_4003349/experiment_data.npy",
    "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_82e86dc616f640308f7f4d993670c8a2_proc_4003351/experiment_data.npy",
    "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_3d6daa8cef784a09ba46c81158e45be7_proc_4003350/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(root, rel_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate and plot if data is loaded
if all_experiment_data:
    # Iterate over each dataset (e.g., "synthetic")
    for dataset in all_experiment_data[0].keys():
        losses_train_runs, losses_val_runs = [], []
        align_train_runs, align_val_runs = [], []
        accuracy_runs = []
        epochs = None

        # Collect metrics from each run
        for exp_data in all_experiment_data:
            exp = exp_data.get(dataset, {})
            if not exp:
                continue
            epochs = np.array(exp["epochs"])
            losses_train_runs.append(np.array(exp["losses"]["train"]))
            losses_val_runs.append(np.array(exp["losses"]["val"]))
            align_train_runs.append(np.array(exp["metrics"]["train"]))
            align_val_runs.append(np.array(exp["metrics"]["val"]))
            preds = np.array(exp["predictions"])
            gts = np.array(exp["ground_truth"])
            accuracy_runs.append(np.mean(preds == gts))

        # Convert to arrays and compute mean + SEM
        lt = np.vstack(losses_train_runs)
        lv = np.vstack(losses_val_runs)
        at = np.vstack(align_train_runs)
        av = np.vstack(align_val_runs)
        n = lt.shape[0]
        mean_lt, sem_lt = lt.mean(axis=0), lt.std(axis=0, ddof=1) / np.sqrt(n)
        mean_lv, sem_lv = lv.mean(axis=0), lv.std(axis=0, ddof=1) / np.sqrt(n)
        mean_at, sem_at = at.mean(axis=0), at.std(axis=0, ddof=1) / np.sqrt(n)
        mean_av, sem_av = av.mean(axis=0), av.std(axis=0, ddof=1) / np.sqrt(n)

        # Plot aggregated loss curves
        try:
            plt.figure()
            plt.errorbar(
                epochs,
                mean_lt,
                yerr=sem_lt,
                marker="o",
                capsize=3,
                label="Train Loss Mean",
            )
            plt.errorbar(
                epochs,
                mean_lv,
                yerr=sem_lv,
                marker="s",
                capsize=3,
                label="Val Loss Mean",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dataset} Dataset Aggregated Loss Curves\nTrain vs Validation with SEM"
            )
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_aggregated_loss_sem.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating {dataset} loss plot: {e}")
            plt.close()

        # Plot aggregated alignment curves
        try:
            plt.figure()
            plt.errorbar(
                epochs,
                mean_at,
                yerr=sem_at,
                marker="o",
                capsize=3,
                label="Train Alignment Mean",
            )
            plt.errorbar(
                epochs,
                mean_av,
                yerr=sem_av,
                marker="s",
                capsize=3,
                label="Val Alignment Mean",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Alignment (1 - JSD)")
            plt.title(
                f"{dataset} Dataset Aggregated Alignment Metric\nTrain vs Validation with SEM"
            )
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_aggregated_alignment_sem.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating {dataset} alignment plot: {e}")
            plt.close()

        # Print aggregated accuracy
        acc_array = np.array(accuracy_runs)
        mean_acc = acc_array.mean()
        sem_acc = acc_array.std(ddof=1) / np.sqrt(len(acc_array))
        print(f"Aggregated accuracy on {dataset}: {mean_acc:.4f} ± {sem_acc:.4f}")
