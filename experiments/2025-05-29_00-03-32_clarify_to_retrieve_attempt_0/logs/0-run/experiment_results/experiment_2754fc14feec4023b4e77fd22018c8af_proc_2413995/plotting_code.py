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

for name, ds_info in experiment_data.get("post_clar_noise", {}).items():
    try:
        m = ds_info["metrics"]
        noise = m["noise_levels"]
        baseline = m["baseline_acc"]
        clarified = m["clar_acc"]
        ces = m["CES"]

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(noise, baseline, marker="o", label="Baseline")
        axs[0].plot(noise, clarified, marker="o", label="Clarified")
        axs[0].set_xlabel("Noise Level")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_title("Accuracy Curves (Baseline vs Clarified)")
        axs[0].legend()

        axs[1].plot(noise, ces, marker="o", color="green")
        axs[1].set_xlabel("Noise Level")
        axs[1].set_ylabel("CES")
        axs[1].set_title("Cost-Effectiveness Score (CES)")

        fig.suptitle(f"{name} Dataset Metrics")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(working_dir, f"{name}_noise_metrics.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for {name}: {e}")
        plt.close("all")
