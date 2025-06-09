import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Iterate datasets
for ds, entry in experiment_data.get(
    "multi_domain_synthetic_specification", {}
).items():
    losses = entry.get("losses", {})
    rates = entry.get("metrics", {})
    # Print final metrics
    try:
        print(
            f'Dataset={ds}: Final train rate={rates["train"][-1]:.4f}, final val rate={rates["val"][-1]:.4f}'
        )
    except Exception:
        pass

    # Loss curve
    try:
        plt.figure()
        plt.plot(
            range(1, len(losses.get("train", [])) + 1),
            losses.get("train", []),
            label="Train Loss",
        )
        plt.plot(
            range(1, len(losses.get("val", [])) + 1),
            losses.get("val", []),
            label="Val Loss",
        )
        plt.suptitle(f"Loss Curve for {ds}")
        plt.title("Left: Train Loss, Right: Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds}: {e}")
        plt.close()

    # Accuracy curve
    try:
        plt.figure()
        plt.plot(
            range(1, len(rates.get("train", [])) + 1),
            rates.get("train", []),
            label="Train Rate",
        )
        plt.plot(
            range(1, len(rates.get("val", [])) + 1),
            rates.get("val", []),
            label="Val Rate",
        )
        plt.suptitle(f"Accuracy Curve for {ds}")
        plt.title("Left: Train Rate, Right: Val Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Pass Rate")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {ds}: {e}")
        plt.close()
