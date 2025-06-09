import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load all experiment data
experiment_data_path_list = [
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_1ca653a5762847979dd3440b11e776bd_proc_2379937/experiment_data.npy",
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_d55fa0d5edf8493ab1ab91886ff0b0f9_proc_2379936/experiment_data.npy",
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_0eece47f8b6b401ba4419b77a1ea3244_proc_2379938/experiment_data.npy",
]
all_experiment_data = []
try:
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Extract synthetic XOR results per run
all_runs = []
for exp in all_experiment_data:
    d = exp.get("hidden_layer_size", {}).get("synthetic_xor", {})
    if d:
        all_runs.append(d)
if not all_runs:
    print("No synthetic_xor data found.")
else:
    sizes = all_runs[0].get("sizes", [])

    # Organize curves per run
    loss_tr_runs = [run.get("losses", {}).get("train", []) for run in all_runs]
    loss_val_runs = [run.get("losses", {}).get("val", []) for run in all_runs]
    ces_tr_runs = [run.get("metrics", {}).get("train", []) for run in all_runs]
    ces_val_runs = [run.get("metrics", {}).get("val", []) for run in all_runs]

    # Plot mean loss curves with SEM
    try:
        plt.figure()
        for idx, sz in enumerate(sizes):
            # gather train curves across runs for this size
            train_curves = [lr[idx] for lr in loss_tr_runs if len(lr) > idx and lr[idx]]
            val_curves = [lv[idx] for lv in loss_val_runs if len(lv) > idx and lv[idx]]
            if not train_curves or not val_curves:
                continue
            # align to shortest
            L_tr = min(len(c) for c in train_curves)
            L_val = min(len(c) for c in val_curves)
            arr_tr = np.array([c[:L_tr] for c in train_curves])
            arr_val = np.array([c[:L_val] for c in val_curves])
            mean_tr = arr_tr.mean(axis=0)
            mean_val = arr_val.mean(axis=0)
            sem_tr = (
                arr_tr.std(axis=0, ddof=1) / np.sqrt(arr_tr.shape[0])
                if arr_tr.shape[0] > 1
                else np.zeros_like(mean_tr)
            )
            sem_val = (
                arr_val.std(axis=0, ddof=1) / np.sqrt(arr_val.shape[0])
                if arr_val.shape[0] > 1
                else np.zeros_like(mean_val)
            )
            epochs_tr = np.arange(1, L_tr + 1)
            epochs_val = np.arange(1, L_val + 1)
            plt.plot(epochs_tr, mean_tr, label=f"Train loss mean (size={sz})")
            plt.fill_between(epochs_tr, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
            plt.plot(epochs_val, mean_val, "--", label=f"Val loss mean (size={sz})")
            plt.fill_between(
                epochs_val, mean_val - sem_val, mean_val + sem_val, alpha=0.2
            )
        plt.title("Synthetic XOR: Mean Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_xor_mean_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating mean loss plot: {e}")
        plt.close()

    # Plot mean CES curves with SEM
    try:
        plt.figure()
        for idx, sz in enumerate(sizes):
            train_curves = [cr[idx] for cr in ces_tr_runs if len(cr) > idx and cr[idx]]
            val_curves = [cr[idx] for cr in ces_val_runs if len(cr) > idx and cr[idx]]
            if not train_curves or not val_curves:
                continue
            L_tr = min(len(c) for c in train_curves)
            L_val = min(len(c) for c in val_curves)
            arr_tr = np.array([c[:L_tr] for c in train_curves])
            arr_val = np.array([c[:L_val] for c in val_curves])
            mean_tr = arr_tr.mean(axis=0)
            mean_val = arr_val.mean(axis=0)
            sem_tr = (
                arr_tr.std(axis=0, ddof=1) / np.sqrt(arr_tr.shape[0])
                if arr_tr.shape[0] > 1
                else np.zeros_like(mean_tr)
            )
            sem_val = (
                arr_val.std(axis=0, ddof=1) / np.sqrt(arr_val.shape[0])
                if arr_val.shape[0] > 1
                else np.zeros_like(mean_val)
            )
            epochs_tr = np.arange(1, L_tr + 1)
            epochs_val = np.arange(1, L_val + 1)
            plt.plot(epochs_tr, mean_tr, label=f"Train CES mean (size={sz})")
            plt.fill_between(epochs_tr, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
            plt.plot(epochs_val, mean_val, "--", label=f"Val CES mean (size={sz})")
            plt.fill_between(
                epochs_val, mean_val - sem_val, mean_val + sem_val, alpha=0.2
            )
        plt.title("Synthetic XOR: Mean Training vs Validation CES")
        plt.xlabel("Epoch")
        plt.ylabel("CES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_xor_mean_CES_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating mean CES plot: {e}")
        plt.close()

    # Plot final validation CES bar chart (mean ± SEM)
    try:
        means = []
        sems = []
        for idx, sz in enumerate(sizes):
            finals = [c[idx][-1] for c in ces_val_runs if len(c) > idx and c[idx]]
            if not finals:
                means.append(0)
                sems.append(0)
            else:
                arr = np.array(finals)
                means.append(arr.mean())
                sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0)
        x = np.arange(len(sizes))
        plt.figure()
        plt.bar(x, means, yerr=sems, capsize=5, label="Final Val CES mean")
        plt.xticks(x, [str(s) for s in sizes])
        plt.title("Synthetic XOR: Final Validation CES by Hidden Size (Mean ± SEM)")
        plt.xlabel("Hidden Layer Size")
        plt.ylabel("CES")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_xor_final_CES_bar_with_error.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating final CES bar plot: {e}")
        plt.close()
