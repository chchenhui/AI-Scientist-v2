import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Generate plots per dataset
for name, data in experiment_data.items():
    try:
        noise = data["noise_levels"]
        baseline = data["baseline_acc"]
        clar = data["clar_acc"]
        gain = data["AccuracyGainPerClarificationTurn"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f"{name} Metrics vs Noise Levels\nLeft: Accuracy, Right: Accuracy Gain per Clarification Turn"
        )
        # Left: Accuracy curves
        axes[0].plot(noise, baseline, marker="o", label="Baseline")
        axes[0].plot(noise, clar, marker="s", label="Clarification")
        axes[0].set_xlabel("Noise Level")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy vs Noise Levels")
        axes[0].legend()
        # Right: Gain per turn
        axes[1].plot(noise, gain, marker="d", color="C2")
        axes[1].set_xlabel("Noise Level")
        axes[1].set_ylabel("Accuracy Gain per Turn")
        axes[1].set_title("Accuracy Gain per Clarification Turn")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, f"{name}_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {name} metrics plot: {e}")
        plt.close()
