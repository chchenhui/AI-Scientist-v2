import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
else:
    for name, metrics in experiment_data.get("retrieval_size", {}).items():
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            # Top-left: Baseline Accuracy
            axes[0, 0].plot(metrics["k"], metrics["baseline_acc"], marker="o")
            axes[0, 0].set_xlabel("Retrieval Size k")
            axes[0, 0].set_ylabel("Accuracy")
            axes[0, 0].set_title("Baseline Accuracy")
            # Top-right: Clarified Accuracy
            axes[0, 1].plot(metrics["k"], metrics["clar_acc"], marker="o")
            axes[0, 1].set_xlabel("Retrieval Size k")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].set_title("Clarified Accuracy")
            # Bottom-left: CES
            axes[1, 0].plot(metrics["k"], metrics["CES"], marker="o")
            axes[1, 0].set_xlabel("Retrieval Size k")
            axes[1, 0].set_ylabel("CES")
            axes[1, 0].set_title("Cost Effectiveness Score")
            # Bottom-right: Avg Turns
            axes[1, 1].plot(metrics["k"], metrics["avg_turns"], marker="o")
            axes[1, 1].set_xlabel("Retrieval Size k")
            axes[1, 1].set_ylabel("Average Turns")
            axes[1, 1].set_title("Average Turns per Query")
            fig.suptitle(
                f"{name} Metrics vs Retrieval Size "
                "(Top Left: Baseline, Top Right: Clarified; "
                "Bottom Left: CES; Bottom Right: Avg Turns)"
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(
                working_dir, f"{name}_metrics_vs_retrieval_size.png"
            )
            fig.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error creating plot for {name}: {e}")
            plt.close("all")
