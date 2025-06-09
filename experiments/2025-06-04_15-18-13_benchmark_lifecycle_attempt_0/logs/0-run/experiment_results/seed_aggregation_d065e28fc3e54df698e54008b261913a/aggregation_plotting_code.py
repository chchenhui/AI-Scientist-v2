import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_85e09a59c15944daa4beb064e125da71_proc_3746533/experiment_data.npy",
    "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_70f6303a6a5b4df7b88396bd2e8f6b58_proc_3746534/experiment_data.npy",
    "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_69bd0334c8434f93b1b1ef5b4624c546_proc_3746535/experiment_data.npy",
]

# Load all experiment runs
try:
    all_runs = []
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        all_runs.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate per-dataset, per-model metrics
data_agg = {}
for ds in all_runs[0].keys():
    data_agg[ds] = {"metrics": {}, "discrimination_score": None}
    # Stack discrimination_score
    disc_stack = np.array([run[ds]["discrimination_score"] for run in all_runs])
    data_agg[ds]["discrimination_score_mean"] = disc_stack.mean(axis=0)
    data_agg[ds]["discrimination_score_sem"] = disc_stack.std(axis=0) / np.sqrt(
        disc_stack.shape[0]
    )
    # Stack metrics per model
    for model in all_runs[0][ds]["metrics"].keys():
        tr_stack = np.array(
            [run[ds]["metrics"][model]["train_loss"] for run in all_runs]
        )
        vl_stack = np.array([run[ds]["metrics"][model]["val_loss"] for run in all_runs])
        va_stack = np.array([run[ds]["metrics"][model]["val_acc"] for run in all_runs])
        data_agg[ds]["metrics"][model] = {
            "train_loss_mean": tr_stack.mean(axis=0),
            "train_loss_sem": tr_stack.std(axis=0) / np.sqrt(tr_stack.shape[0]),
            "val_loss_mean": vl_stack.mean(axis=0),
            "val_loss_sem": vl_stack.std(axis=0) / np.sqrt(vl_stack.shape[0]),
            "val_acc_mean": va_stack.mean(axis=0),
            "val_acc_sem": va_stack.std(axis=0) / np.sqrt(va_stack.shape[0]),
        }

# Print aggregated final validation accuracies
for ds, info in data_agg.items():
    for model, m in info["metrics"].items():
        mean_acc = m["val_acc_mean"][-1]
        sem_acc = m["val_acc_sem"][-1]
        print(f"{ds} - {model}: mean final val_acc = {mean_acc:.4f} ± {sem_acc:.4f}")

# Plot aggregated loss and accuracy curves per dataset
for ds, info in data_agg.items():
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        epochs = np.arange(
            1, len(next(iter(info["metrics"].values()))["train_loss_mean"]) + 1
        )
        for model, m in info["metrics"].items():
            ax1.errorbar(
                epochs,
                m["train_loss_mean"],
                yerr=m["train_loss_sem"],
                label=f"{model} train",
                capsize=3,
            )
            ax1.errorbar(
                epochs,
                m["val_loss_mean"],
                yerr=m["val_loss_sem"],
                linestyle="--",
                label=f"{model} val",
                capsize=3,
            )
            ax2.errorbar(
                epochs, m["val_acc_mean"], yerr=m["val_acc_sem"], label=model, capsize=3
            )
        fig.suptitle(
            f"{ds.capitalize()} Aggregated Metrics (Left: Loss Curves, Right: Accuracy Curves)"
        )
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2.set_title("Accuracy Curves")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        plt.savefig(
            os.path.join(working_dir, f"{ds}_aggregated_loss_accuracy_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot for {ds}: {e}")
        plt.close()

# Plot aggregated discrimination score across datasets
try:
    plt.figure()
    for ds, info in data_agg.items():
        epochs = np.arange(1, len(info["discrimination_score_mean"]) + 1)
        plt.errorbar(
            epochs,
            info["discrimination_score_mean"],
            yerr=info["discrimination_score_sem"],
            label=ds,
            capsize=3,
        )
    plt.title("Aggregated Discrimination Score Across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Discrimination Score")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "aggregated_discrimination_score_across_datasets.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated discrimination score plot: {e}")
    plt.close()

# Plot aggregated final validation accuracy comparison across datasets
try:
    labels = list(data_agg.keys())
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    models = list(next(iter(data_agg.values()))["metrics"].keys())
    for i, model in enumerate(models):
        means = [data_agg[ds]["metrics"][model]["val_acc_mean"][-1] for ds in labels]
        sems = [data_agg[ds]["metrics"][model]["val_acc_sem"][-1] for ds in labels]
        ax.bar(x + i * width, means, width, yerr=sems, capsize=3, label=model)
    ax.set_title("Aggregated Final Validation Accuracy Comparison Across Datasets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(
        os.path.join(working_dir, "aggregated_final_val_accuracy_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final accuracy comparison plot: {e}")
    plt.close()
