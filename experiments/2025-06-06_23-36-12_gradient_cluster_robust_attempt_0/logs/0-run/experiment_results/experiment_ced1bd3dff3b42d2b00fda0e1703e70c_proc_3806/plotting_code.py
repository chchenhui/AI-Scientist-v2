import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
else:
    exp = data["warmup_epochs"]["synthetic"]
    warmup = exp["warmup_values"]
    train_losses = exp["losses"]["train"]
    val_losses = exp["losses"]["val"]
    train_metrics = exp["metrics"]["train"]
    val_metrics = exp["metrics"]["val"]

    try:
        plt.figure()
        for i, w in enumerate(warmup):
            plt.plot(train_losses[i], label=f"warmup={w}")
        for i, w in enumerate(warmup):
            plt.plot(val_losses[i], linestyle="--", label=f"warmup={w}")
        plt.title("Synthetic: Loss Curves (train solid, val dashed)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    try:
        plt.figure()
        for i, w in enumerate(warmup):
            plt.plot(train_metrics[i], label=f"warmup={w}")
        for i, w in enumerate(warmup):
            plt.plot(val_metrics[i], linestyle="--", label=f"warmup={w}")
        plt.title("Synthetic: WG Accuracy Curves (train solid, val dashed)")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted‐Group Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_wg_acc_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

    try:
        final_val = [m[-1] for m in val_metrics]
        plt.figure()
        plt.bar([str(w) for w in warmup], final_val)
        plt.title("Synthetic: Final Validation WG Accuracy")
        plt.xlabel("Warmup Epochs")
        plt.ylabel("WG Accuracy")
        plt.savefig(os.path.join(working_dir, "synthetic_final_val_wg_acc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final bar plot: {e}")
        plt.close()
