import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "None/experiment_data.npy",
        "None/experiment_data.npy",
        "experiments/2025-06-07_15-26-52_abstract_interpretation_guided_generation_attempt_0/logs/0-run/experiment_results/experiment_2160233ef98d4359bd56051df6737eb2_proc_87822/experiment_data.npy",
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

try:
    d_list = [
        exp["classification_head_depth"]["synthetic"] for exp in all_experiment_data
    ]
    head_depths = d_list[0]["head_depths"]
    plt.figure()
    for idx, hd in enumerate(head_depths):
        train_arrs = [d["losses"]["train"][idx] for d in d_list]
        val_arrs = [d["losses"]["val"][idx] for d in d_list]
        min_epochs = min(arr.shape[0] for arr in train_arrs + val_arrs)
        train_stack = np.vstack([arr[:min_epochs] for arr in train_arrs])
        val_stack = np.vstack([arr[:min_epochs] for arr in val_arrs])
        epochs = np.arange(1, min_epochs + 1)
        train_mean = train_stack.mean(axis=0)
        train_sem = train_stack.std(axis=0, ddof=1) / np.sqrt(train_stack.shape[0])
        val_mean = val_stack.mean(axis=0)
        val_sem = val_stack.std(axis=0, ddof=1) / np.sqrt(val_stack.shape[0])
        plt.plot(epochs, train_mean, label=f"Train Loss Depth {hd}", linestyle="-")
        plt.fill_between(
            epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.2
        )
        plt.plot(epochs, val_mean, label=f"Val Loss Depth {hd}", linestyle="--")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)
    plt.suptitle("Aggregated Loss Curves: Synthetic Dataset")
    plt.title("Mean ± SEM across experiments for Train (solid) and Val (dashed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

try:
    plt.figure()
    for idx, hd in enumerate(head_depths):
        train_arrs = [d["metrics"]["train"][idx] for d in d_list]
        val_arrs = [d["metrics"]["val"][idx] for d in d_list]
        min_epochs = min(arr.shape[0] for arr in train_arrs + val_arrs)
        train_stack = np.vstack([arr[:min_epochs] for arr in train_arrs])
        val_stack = np.vstack([arr[:min_epochs] for arr in val_arrs])
        epochs = np.arange(1, min_epochs + 1)
        train_mean = train_stack.mean(axis=0)
        train_sem = train_stack.std(axis=0, ddof=1) / np.sqrt(train_stack.shape[0])
        val_mean = val_stack.mean(axis=0)
        val_sem = val_stack.std(axis=0, ddof=1) / np.sqrt(val_stack.shape[0])
        plt.plot(epochs, train_mean, label=f"Train AICR Depth {hd}", linestyle="-")
        plt.fill_between(
            epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.2
        )
        plt.plot(epochs, val_mean, label=f"Val AICR Depth {hd}", linestyle="--")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)
    plt.suptitle("Aggregated AICR Metrics: Synthetic Dataset")
    plt.title("Mean ± SEM across experiments for Train (solid) and Val (dashed)")
    plt.xlabel("Epoch")
    plt.ylabel("AICR Rate")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_aicr_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated AICR plot: {e}")
    plt.close()

try:
    plt.figure()
    val_final_means = []
    val_final_sems = []
    for idx, hd in enumerate(head_depths):
        train_arrs = [d["classification_accuracy"]["train"][idx] for d in d_list]
        val_arrs = [d["classification_accuracy"]["val"][idx] for d in d_list]
        min_epochs = min(arr.shape[0] for arr in train_arrs + val_arrs)
        train_stack = np.vstack([arr[:min_epochs] for arr in train_arrs])
        val_stack = np.vstack([arr[:min_epochs] for arr in val_arrs])
        epochs = np.arange(1, min_epochs + 1)
        train_mean = train_stack.mean(axis=0)
        train_sem = train_stack.std(axis=0, ddof=1) / np.sqrt(train_stack.shape[0])
        val_mean = val_stack.mean(axis=0)
        val_sem = val_stack.std(axis=0, ddof=1) / np.sqrt(val_stack.shape[0])
        val_final_means.append(val_mean[-1])
        val_final_sems.append(val_sem[-1])
        plt.plot(epochs, train_mean, label=f"Train Acc Depth {hd}", linestyle="-")
        plt.fill_between(
            epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.2
        )
        plt.plot(epochs, val_mean, label=f"Val Acc Depth {hd}", linestyle="--")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)
    plt.suptitle("Aggregated Classification Accuracy: Synthetic Dataset")
    plt.title("Mean ± SEM across experiments for Train (solid) and Val (dashed)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_aggregated.png"))
    plt.close()
    for hd, fm, fs in zip(head_depths, val_final_means, val_final_sems):
        print(f"Synthetic Depth {hd}: Final Val Acc = {fm:.3f} ± {fs:.3f}")
except Exception as e:
    print(f"Error creating aggregated classification accuracy plot: {e}")
    plt.close()
