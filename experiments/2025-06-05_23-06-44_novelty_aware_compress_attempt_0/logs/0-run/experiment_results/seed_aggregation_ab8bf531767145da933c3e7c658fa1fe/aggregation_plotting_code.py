import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_ba6e81c792e548a1bcdd2ff8bdfa40e3_proc_3959671/experiment_data.npy",
    "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_4976fc12f69f448f87939dd52a86af45_proc_3959669/experiment_data.npy",
    "experiments/2025-06-05_23-06-44_novelty_aware_compress_attempt_0/logs/0-run/experiment_results/experiment_1c49ea7bd397430d999d7adb1af4b134_proc_3959670/experiment_data.npy",
]

# Load all runs
try:
    all_experiment_data = []
    for exp_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), exp_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

dataset_names = list(all_experiment_data[0].keys()) if all_experiment_data else []

# Print aggregated final metrics
try:
    for ds in dataset_names:
        train_loss = np.array(
            [r[ds]["losses"]["train"][-1] for r in all_experiment_data]
        )
        val_loss = np.array([r[ds]["losses"]["val"][-1] for r in all_experiment_data])
        train_ret = np.array(
            [
                r[ds]["metrics"]["Memory Retention Ratio"]["train"][-1]
                for r in all_experiment_data
            ]
        )
        val_ret = np.array(
            [
                r[ds]["metrics"]["Memory Retention Ratio"]["val"][-1]
                for r in all_experiment_data
            ]
        )
        train_eme = np.array(
            [
                r[ds]["metrics"]["Entropy-Weighted Memory Efficiency"]["train"][-1]
                for r in all_experiment_data
            ]
        )
        val_eme = np.array(
            [
                r[ds]["metrics"]["Entropy-Weighted Memory Efficiency"]["val"][-1]
                for r in all_experiment_data
            ]
        )

        def fmt(arr):
            m = arr.mean()
            sem = arr.std(ddof=1) / np.sqrt(len(arr))
            return f"{m:.4f} ± {sem:.4f}"

        print(f"{ds} Final Train Loss: {fmt(train_loss)}, Val Loss: {fmt(val_loss)}")
        print(
            f"{ds} Final Train Retention: {fmt(train_ret)}, Val Retention: {fmt(val_ret)}"
        )
        print(f"{ds} Final Train EME: {fmt(train_eme)}, Val EME: {fmt(val_eme)}")
except Exception as e:
    print(f"Error printing aggregated final metrics: {e}")

# Mean Loss Curves with SEM
try:
    plt.figure()
    for ds in dataset_names:
        t_list = [r[ds]["losses"]["train"] for r in all_experiment_data]
        v_list = [r[ds]["losses"]["val"] for r in all_experiment_data]
        min_ep = min(len(x) for x in t_list)
        t_arr = np.array([x[:min_ep] for x in t_list])
        v_arr = np.array([x[:min_ep] for x in v_list])
        ep = np.arange(min_ep)
        t_mean, t_sem = t_arr.mean(0), t_arr.std(ddof=1, axis=0) / np.sqrt(
            t_arr.shape[0]
        )
        v_mean, v_sem = v_arr.mean(0), v_arr.std(ddof=1, axis=0) / np.sqrt(
            v_arr.shape[0]
        )
        plt.plot(ep, t_mean, label=f"{ds} train mean")
        plt.fill_between(ep, t_mean - t_sem, t_mean + t_sem, alpha=0.2)
        plt.plot(ep, v_mean, "--", label=f"{ds} val mean")
        plt.fill_between(ep, v_mean - v_sem, v_mean + v_sem, alpha=0.2)
    plt.title("Mean Loss Curves Across Runs\nDatasets: " + ", ".join(dataset_names))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_mean_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss curves plot: {e}")
    plt.close()

# Mean Memory Retention Ratio with SEM
try:
    plt.figure()
    for ds in dataset_names:
        t_list = [
            r[ds]["metrics"]["Memory Retention Ratio"]["train"]
            for r in all_experiment_data
        ]
        v_list = [
            r[ds]["metrics"]["Memory Retention Ratio"]["val"]
            for r in all_experiment_data
        ]
        min_ep = min(len(x) for x in t_list)
        t_arr = np.array([x[:min_ep] for x in t_list])
        v_arr = np.array([x[:min_ep] for x in v_list])
        ep = np.arange(min_ep)
        t_mean, t_sem = t_arr.mean(0), t_arr.std(ddof=1, axis=0) / np.sqrt(
            t_arr.shape[0]
        )
        v_mean, v_sem = v_arr.mean(0), v_arr.std(ddof=1, axis=0) / np.sqrt(
            v_arr.shape[0]
        )
        plt.plot(ep, t_mean, label=f"{ds} train mean")
        plt.fill_between(ep, t_mean - t_sem, t_mean + t_sem, alpha=0.2)
        plt.plot(ep, v_mean, "--", label=f"{ds} val mean")
        plt.fill_between(ep, v_mean - v_sem, v_mean + v_sem, alpha=0.2)
    plt.title(
        "Mean Memory Retention Ratio Across Runs\nDatasets: " + ", ".join(dataset_names)
    )
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_mean_retention_ratios.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean retention ratio plot: {e}")
    plt.close()

# Mean Entropy-Weighted Memory Efficiency with SEM
try:
    plt.figure()
    for ds in dataset_names:
        t_list = [
            r[ds]["metrics"]["Entropy-Weighted Memory Efficiency"]["train"]
            for r in all_experiment_data
        ]
        v_list = [
            r[ds]["metrics"]["Entropy-Weighted Memory Efficiency"]["val"]
            for r in all_experiment_data
        ]
        min_ep = min(len(x) for x in t_list)
        t_arr = np.array([x[:min_ep] for x in t_list])
        v_arr = np.array([x[:min_ep] for x in v_list])
        ep = np.arange(min_ep)
        t_mean, t_sem = t_arr.mean(0), t_arr.std(ddof=1, axis=0) / np.sqrt(
            t_arr.shape[0]
        )
        v_mean, v_sem = v_arr.mean(0), v_arr.std(ddof=1, axis=0) / np.sqrt(
            v_arr.shape[0]
        )
        plt.plot(ep, t_mean, label=f"{ds} train mean")
        plt.fill_between(ep, t_mean - t_sem, t_mean + t_sem, alpha=0.2)
        plt.plot(ep, v_mean, "--", label=f"{ds} val mean")
        plt.fill_between(ep, v_mean - v_sem, v_mean + v_sem, alpha=0.2)
    plt.title(
        "Mean Entropy-Weighted Memory Efficiency Across Runs\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.xlabel("Epoch")
    plt.ylabel("Entropy-Weighted Memory Efficiency")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "all_datasets_mean_entropy_weighted_memory_efficiency.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean entropy-weighted memory efficiency plot: {e}")
    plt.close()
