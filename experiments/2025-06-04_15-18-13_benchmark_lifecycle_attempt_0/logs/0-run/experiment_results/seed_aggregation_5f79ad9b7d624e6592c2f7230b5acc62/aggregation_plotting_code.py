import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Paths to all experiment_data.npy files
exp_paths = [
    "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_f41f33d4aedb4e4d9add6d07ea1eb8fb_proc_3707563/experiment_data.npy",
    "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_bd4200573eac4c5b88b8cc01b63cfa1d_proc_3707565/experiment_data.npy",
    "None/experiment_data.npy",
]

# Load all experiment data
all_data = []
for path in exp_paths:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        data = np.load(full_path, allow_pickle=True).item()
        all_data.append(data)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Determine all n_epochs values and all model names
n_epochs_list = sorted(
    {n for d in all_data for n in d.get("n_epochs", {})}, key=lambda x: int(x)
)
models = sorted(
    {
        m
        for d in all_data
        for n in d.get("n_epochs", {})
        for m in d["n_epochs"][n]["models"]
    }
)

# Plot mean final original accuracy with SE bars
try:
    x = np.arange(len(n_epochs_list))
    width = 0.8 / len(models)
    plt.figure()
    for idx, model in enumerate(models):
        means, ses = [], []
        for n in n_epochs_list:
            vals = [
                d["n_epochs"][n]["models"][model]["metrics"]["orig_acc"][-1]
                for d in all_data
                if n in d.get("n_epochs", {})
            ]
            means.append(np.mean(vals))
            ses.append(np.std(vals) / np.sqrt(len(vals)))
            print(
                f"{model} (n_epochs={n}): mean_orig_acc={means[-1]:.4f} +/- {ses[-1]:.4f}"
            )
        plt.bar(x + idx * width, means, width, yerr=ses, label=f"{model} orig_acc")
    plt.xlabel("n_epochs")
    plt.ylabel("Accuracy")
    plt.title("Mean Final Original Accuracy with SE Bars (MNIST)")
    plt.xticks(x + width * (len(models) - 1) / 2, n_epochs_list)
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mnist_mean_orig_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating orig accuracy bar plot: {e}")
    plt.close()

# Plot mean final augmented accuracy with SE bars
try:
    x = np.arange(len(n_epochs_list))
    width = 0.8 / len(models)
    plt.figure()
    for idx, model in enumerate(models):
        means, ses = [], []
        for n in n_epochs_list:
            vals = [
                d["n_epochs"][n]["models"][model]["metrics"]["aug_acc"][-1]
                for d in all_data
                if n in d.get("n_epochs", {})
            ]
            means.append(np.mean(vals))
            ses.append(np.std(vals) / np.sqrt(len(vals)))
            print(
                f"{model} (n_epochs={n}): mean_aug_acc={means[-1]:.4f} +/- {ses[-1]:.4f}"
            )
        plt.bar(x + idx * width, means, width, yerr=ses, label=f"{model} aug_acc")
    plt.xlabel("n_epochs")
    plt.ylabel("Accuracy")
    plt.title("Mean Final Augmented Accuracy with SE Bars (MNIST)")
    plt.xticks(x + width * (len(models) - 1) / 2, n_epochs_list)
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mnist_mean_aug_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aug accuracy bar plot: {e}")
    plt.close()

# Plot aggregated training vs val loss for each n_epochs (MLP & CNN)
for n in n_epochs_list:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for idx, model in enumerate(models):
            # collect losses across runs
            tr_list = [
                d["n_epochs"][n]["models"][model]["losses"]["train"]
                for d in all_data
                if n in d.get("n_epochs", {})
            ]
            val_list = [
                d["n_epochs"][n]["models"][model]["losses"]["val"]
                for d in all_data
                if n in d.get("n_epochs", {})
            ]
            arr_tr = np.vstack(tr_list)
            arr_val = np.vstack(val_list)
            mean_tr = arr_tr.mean(axis=0)
            se_tr = arr_tr.std(axis=0) / np.sqrt(arr_tr.shape[0])
            mean_val = arr_val.mean(axis=0)
            se_val = arr_val.std(axis=0) / np.sqrt(arr_val.shape[0])
            epochs = np.arange(1, len(mean_tr) + 1)
            ax = axes[idx]
            ax.plot(epochs, mean_tr, label="Mean Train Loss")
            ax.fill_between(epochs, mean_tr - se_tr, mean_tr + se_tr, alpha=0.3)
            ax.plot(epochs, mean_val, label="Mean Val Loss")
            ax.fill_between(epochs, mean_val - se_val, mean_val + se_val, alpha=0.3)
            side = "Left" if idx == 0 else "Right"
            ax.set_title(f"{side}: Mean Train vs Val Loss ({model})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.suptitle(f"MNIST Mean Loss Curves with SE (n_epochs={n})")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"mnist_mean_loss_n_epochs_{n}.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating aggregated loss plot for n_epochs={n}: {e}")
        plt.close()

# Plot aggregated CGR vs epoch for all n_epochs
try:
    plt.figure()
    for n in n_epochs_list:
        cgr_list = [
            d["n_epochs"][n]["cgr"] for d in all_data if n in d.get("n_epochs", {})
        ]
        arr = np.vstack(cgr_list)
        mean_cgr = arr.mean(axis=0)
        se_cgr = arr.std(axis=0) / np.sqrt(arr.shape[0])
        epochs = np.arange(1, len(mean_cgr) + 1)
        plt.plot(epochs, mean_cgr, marker="o", label=f"n_epochs={n}")
        plt.fill_between(epochs, mean_cgr - se_cgr, mean_cgr + se_cgr, alpha=0.3)
    plt.title("Mean CGR vs Epoch for MNIST with SE")
    plt.xlabel("Epoch")
    plt.ylabel("CGR")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mnist_mean_cgr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CGR plot: {e}")
    plt.close()
