import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths to experiment_data.npy files
experiment_data_path_list = [
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_c038abef030e46da87b56a9cbbe732c0_proc_231371/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_ecfb2002466e4b95826cdeb685b21f06_proc_231370/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_a825ee1395bc4f27b1629a42b159505b_proc_231372/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data {path}: {e}")

if not all_experiment_data:
    print("No experiment data loaded, exiting.")
else:
    # Utility for Spearman correlation
    def spearman_corr(a, b):
        a_rank = np.argsort(np.argsort(a))
        b_rank = np.argsort(np.argsort(b))
        return np.corrcoef(a_rank, b_rank)[0, 1]

    # Collect metrics across runs
    train_list, val_list, corr_list = [], [], []
    for exp in all_experiment_data:
        try:
            tr = np.array(exp["synthetic"]["metrics"]["train"])
            vl = np.array(exp["synthetic"]["metrics"]["val"])
            preds = exp["synthetic"]["predictions"]
            truths = exp["synthetic"]["ground_truth"]
            cr = np.array([spearman_corr(p, t) for p, t in zip(preds, truths)])
            train_list.append(tr)
            val_list.append(vl)
            corr_list.append(cr)
        except KeyError as e:
            print(f"Missing key in experiment data: {e}")

    # Align lengths
    min_epochs = min(len(x) for x in train_list)
    train_arr = np.vstack([x[:min_epochs] for x in train_list])
    val_arr = np.vstack([x[:min_epochs] for x in val_list])
    corr_arr = np.vstack([x[:min_epochs] for x in corr_list])
    epochs = np.arange(1, min_epochs + 1)

    # Compute mean and standard error
    n_runs = train_arr.shape[0]
    train_mean = train_arr.mean(axis=0)
    train_se = (
        train_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(train_mean)
    )
    val_mean = val_arr.mean(axis=0)
    val_se = (
        val_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(val_mean)
    )
    corr_mean = corr_arr.mean(axis=0)
    corr_se = (
        corr_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(corr_mean)
    )

    # Plot aggregated loss curves
    try:
        plt.figure()
        plt.errorbar(epochs, train_mean, yerr=train_se, label="Train Loss Mean")
        plt.errorbar(epochs, val_mean, yerr=val_se, label="Validation Loss Mean")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Synthetic Dataset: Mean Training and Validation Loss (with SE)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curve_mean_se.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve: {e}")
        plt.close()

    # Plot aggregated Spearman correlation curves
    try:
        plt.figure()
        plt.errorbar(
            epochs, corr_mean, yerr=corr_se, marker="o", label="Spearman Corr Mean"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Spearman Correlation")
        plt.title("Synthetic Dataset: Mean Spearman Correlation per Epoch (with SE)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_spearman_corr_mean_se.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated correlation curve: {e}")
        plt.close()
