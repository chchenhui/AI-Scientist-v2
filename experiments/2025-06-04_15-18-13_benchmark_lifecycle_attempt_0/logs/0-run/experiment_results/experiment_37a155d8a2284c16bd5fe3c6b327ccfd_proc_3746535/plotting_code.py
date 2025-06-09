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

for key_d, d_data in experiment_data.get("network_depth", {}).items():
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        eps_keys = sorted(d_data.keys(), key=lambda x: float(x.split("_")[1]))
        for key_e in eps_keys:
            eps_val = key_e.split("_")[1]
            epochs = np.arange(1, len(d_data[key_e]["losses"]["train"]) + 1)
            # Loss curves
            axs[0].plot(
                epochs,
                d_data[key_e]["losses"]["train"],
                marker="o",
                label=f"ε={eps_val} train",
            )
            axs[0].plot(
                epochs,
                d_data[key_e]["losses"]["val"],
                linestyle="--",
                marker="x",
                label=f"ε={eps_val} val",
            )
            # Accuracy curves
            axs[1].plot(
                epochs,
                d_data[key_e]["metrics"]["orig_acc"],
                marker="o",
                label=f"ε={eps_val} orig",
            )
            axs[1].plot(
                epochs,
                d_data[key_e]["metrics"]["aug_acc"],
                linestyle="--",
                marker="x",
                label=f"ε={eps_val} aug",
            )
        axs[0].set_title("Training and Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[1].set_title("Original and Augmented Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        fig.suptitle(
            f"{key_d} Performance on MNIST (Left: Loss curves, Right: Accuracy curves)"
        )
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"mnist_{key_d}_performance.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {key_d}: {e}")
        plt.close()
