import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
beta1_list = [0.5, 0.7, 0.9, 0.99]

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# iterate over synthetic datasets
for name, ed in experiment_data.get("multi_synthetic", {}).items():
    try:
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(f"Multi-Synthetic {name} Results")
        # Left: Errors
        ax1 = fig.add_subplot(1, 2, 1)
        for i, (tr_err, val_err) in enumerate(
            zip(ed["metrics"]["train"], ed["metrics"]["val"])
        ):
            beta = beta1_list[i]
            ax1.plot(range(1, len(tr_err) + 1), tr_err, label=f"β1={beta} train")
            ax1.plot(
                range(1, len(val_err) + 1),
                val_err,
                linestyle="--",
                label=f"β1={beta} val",
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Relative Error")
        ax1.set_title("Left: Training & Validation Error")
        ax1.legend()
        # Right: Losses
        ax2 = fig.add_subplot(1, 2, 2)
        for i, (tr_loss, val_loss) in enumerate(
            zip(ed["losses"]["train"], ed["losses"]["val"])
        ):
            beta = beta1_list[i]
            ax2.plot(range(1, len(tr_loss) + 1), tr_loss, label=f"β1={beta} train")
            ax2.plot(
                range(1, len(val_loss) + 1),
                val_loss,
                linestyle="--",
                label=f"β1={beta} val",
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE Loss")
        ax2.set_title("Right: Training & Validation Loss")
        ax2.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"multi_synthetic_{name}_error_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating multi_synthetic_{name}_error_loss plot: {e}")
        plt.close()
