import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Determine dataset names and ablation types
if experiment_data:
    ablation_types = list(experiment_data.keys())
    dataset_names = list(next(iter(experiment_data.values())).keys())
else:
    ablation_types, dataset_names = [], []

for dataset in dataset_names:
    try:
        # Collect metrics across ablation types
        baseline_accs, clar_accs, avg_turnss, ces_s = [], [], [], []
        for ab in ablation_types:
            m = experiment_data[ab][dataset]["metrics"]
            baseline_accs.append(m["baseline_acc"])
            clar_accs.append(m["clar_acc"])
            avg_turnss.append(m["avg_turns"])
            ces_s.append(m["CES"])

        # Build figure with three subplots
        fig, axes = plt.subplots(3, 1, figsize=(6, 12))
        fig.suptitle(f"Dataset: {dataset}")

        x = np.arange(len(ablation_types))
        width = 0.35
        axes[0].bar(x - width / 2, baseline_accs, width, label="Baseline")
        axes[0].bar(x + width / 2, clar_accs, width, label="Clarification")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ablation_types, rotation=45)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(
            "Accuracy Comparison\nLeft: Baseline Accuracy, Right: Clarification Accuracy"
        )
        axes[0].legend()

        axes[1].bar(ablation_types, avg_turnss, color="gray")
        axes[1].set_ylabel("Average Turns")
        axes[1].set_title("Average Turns per Question")

        axes[2].bar(ablation_types, ces_s, color="orange")
        axes[2].set_ylabel("CES")
        axes[2].set_title("Communication Efficiency Score (CES)")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(
            working_dir, f"{dataset.replace('/', '_')}_metrics_summary.png"
        )
        fig.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error creating {dataset} metrics summary plot: {e}")
        plt.close()
