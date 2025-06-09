import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths to the three replicates
experiment_data_path_list = [
    "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_07421c1465bc4311a832a0670685bf7c_proc_3980098/experiment_data.npy",
    "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_5fc1910ac4d34fbdad984d95ba748f5d_proc_3980098/experiment_data.npy",
    "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_f22cf116ae2045f98cb77a0b0fb55bf1_proc_3980098/experiment_data.npy",
]

# Load experiment data from all replicates
all_experiments = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiments.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiments:
    print("No data loaded, exiting.")
else:
    # Use the first replicate to get dataset and metric keys
    for ds in all_experiments[0].keys():
        # Aggregate losses
        try:
            train_losses = []
            val_losses = []
            for exp in all_experiments:
                ds_data = exp.get(ds, {})
                train_losses.append(ds_data["losses"]["train"])
                val_losses.append(ds_data["losses"]["val"])
            train_losses = np.array(train_losses)
            val_losses = np.array(val_losses)
            epochs = np.arange(1, train_losses.shape[1] + 1)
            mean_tr = train_losses.mean(axis=0)
            sem_tr = train_losses.std(axis=0) / np.sqrt(train_losses.shape[0])
            mean_va = val_losses.mean(axis=0)
            sem_va = val_losses.std(axis=0) / np.sqrt(val_losses.shape[0])

            plt.figure()
            plt.suptitle(f"{ds} - Aggregated over {len(all_experiments)} runs")
            plt.title("Loss over Epochs with SEM")
            plt.plot(epochs, mean_tr, marker="o", label="Train Mean")
            plt.fill_between(epochs, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.3)
            plt.plot(epochs, mean_va, marker="o", linestyle="--", label="Val Mean")
            plt.fill_between(epochs, mean_va - sem_va, mean_va + sem_va, alpha=0.3)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = f"{ds}_aggregate_loss_sem.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {ds}: {e}")
            plt.close()

        # Aggregate each metric
        for metric, splits in all_experiments[0][ds].get("metrics", {}).items():
            try:
                train_vals = []
                val_vals = []
                for exp in all_experiments:
                    m = exp[ds]["metrics"][metric]
                    train_vals.append(m["train"])
                    val_vals.append(m["val"])
                train_vals = np.array(train_vals)
                val_vals = np.array(val_vals)
                epochs = np.arange(1, train_vals.shape[1] + 1)
                mean_trm = train_vals.mean(axis=0)
                sem_trm = train_vals.std(axis=0) / np.sqrt(train_vals.shape[0])
                mean_vam = val_vals.mean(axis=0)
                sem_vam = val_vals.std(axis=0) / np.sqrt(val_vals.shape[0])

                plt.figure()
                plt.suptitle(f"{ds} - {metric} Aggregated")
                plt.title(f"{metric} over Epochs with SEM")
                plt.plot(epochs, mean_trm, marker="o", label="Train Mean")
                plt.fill_between(
                    epochs, mean_trm - sem_trm, mean_trm + sem_trm, alpha=0.3
                )
                plt.plot(epochs, mean_vam, marker="o", linestyle="--", label="Val Mean")
                plt.fill_between(
                    epochs, mean_vam - sem_vam, mean_vam + sem_vam, alpha=0.3
                )
                plt.xlabel("Epoch")
                plt.ylabel(metric)
                plt.legend()
                metric_clean = metric.lower().replace(" ", "_")
                fname = f"{ds}_aggregate_{metric_clean}_sem.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating aggregated {metric} plot for {ds}: {e}")
                plt.close()
