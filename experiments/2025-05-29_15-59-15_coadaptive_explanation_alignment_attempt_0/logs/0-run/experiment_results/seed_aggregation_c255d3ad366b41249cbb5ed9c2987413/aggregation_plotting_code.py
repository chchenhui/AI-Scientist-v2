import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load all experiment data
try:
    root_dir = os.getenv("AI_SCIENTIST_ROOT", os.getcwd())
    experiment_data_path_list = [
        "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_7825e809747547f8a44853d045bf88a6_proc_2565128/experiment_data.npy",
        "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_14836264e96244ab803c267b614bfeb9_proc_2565127/experiment_data.npy",
        "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_3e554f7f44604e0894bd0e6d91ed7346_proc_2565129/experiment_data.npy",
    ]
    all_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(root_dir, p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Determine common batch_size keys and select up to 5
common_keys = set(all_data[0]["batch_size"].keys())
for d in all_data[1:]:
    common_keys &= set(d["batch_size"].keys())
common_keys = sorted(common_keys)
if len(common_keys) > 5:
    idxs = [int(i * (len(common_keys) - 1) / 4) for i in range(5)]
    selected_keys = [common_keys[i] for i in idxs]
else:
    selected_keys = common_keys

# Plot aggregated curves per batch size
for key in selected_keys:
    try:
        acc_trains, acc_vals = [], []
        loss_trains, loss_vals = [], []
        for d in all_data:
            cfg = d["batch_size"][key]
            acc_trains.append(np.array(cfg["metrics"]["train"]))
            acc_vals.append(np.array(cfg["metrics"]["val"]))
            loss_trains.append(np.array(cfg["losses"]["train"]))
            loss_vals.append(np.array(cfg["losses"]["val"]))
        min_ep = min(len(x) for x in acc_trains)
        acc_trains = np.vstack([x[:min_ep] for x in acc_trains])
        acc_vals = np.vstack([x[:min_ep] for x in acc_vals])
        loss_trains = np.vstack([x[:min_ep] for x in loss_trains])
        loss_vals = np.vstack([x[:min_ep] for x in loss_vals])
        mean_at = acc_trains.mean(axis=0)
        sem_at = acc_trains.std(axis=0, ddof=1) / np.sqrt(acc_trains.shape[0])
        mean_av = acc_vals.mean(axis=0)
        sem_av = acc_vals.std(axis=0, ddof=1) / np.sqrt(acc_vals.shape[0])
        mean_lt = loss_trains.mean(axis=0)
        sem_lt = loss_trains.std(axis=0, ddof=1) / np.sqrt(loss_trains.shape[0])
        mean_lv = loss_vals.mean(axis=0)
        sem_lv = loss_vals.std(axis=0, ddof=1) / np.sqrt(loss_vals.shape[0])
        epochs = np.arange(1, min_ep + 1)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].errorbar(epochs, mean_at, yerr=sem_at, label="Train", capsize=3)
        axs[0].errorbar(epochs, mean_av, yerr=sem_av, label="Validation", capsize=3)
        axs[0].set_title("Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].legend()
        axs[1].errorbar(epochs, mean_lt, yerr=sem_lt, label="Train", capsize=3)
        axs[1].errorbar(epochs, mean_lv, yerr=sem_lv, label="Validation", capsize=3)
        axs[1].set_title("Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].legend()
        fig.suptitle(f"Synthetic binary dataset - batch_size {key} (Aggregated)")
        fname = os.path.join(
            working_dir, f"synthetic_binary_dataset_agg_batch_size_{key}_metrics.png"
        )
        plt.savefig(fname)
        plt.close()
        print(f"batch_size {key}: Final Val Acc = {mean_av[-1]:.4f} ± {sem_av[-1]:.4f}")
    except Exception as e:
        print(f"Error creating aggregated plot for batch_size {key}: {e}")
        plt.close()
