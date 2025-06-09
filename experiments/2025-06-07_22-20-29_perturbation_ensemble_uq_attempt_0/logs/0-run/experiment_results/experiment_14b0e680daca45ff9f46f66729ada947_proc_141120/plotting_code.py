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
    key = "synthetic"
    losses = data[key]["losses"]
    metrics = data[key]["metrics"]
    epochs = list(range(1, len(losses["train"]) + 1))
    print(
        f"Final Train Loss: {losses['train'][-1]:.4f}, Final Val Loss: {losses['val'][-1]:.4f}"
    )
    print(
        f"Final Train AUC: {metrics['train'][-1]:.4f}, Final Val AUC: {metrics['val'][-1]:.4f}"
    )

    try:
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.title("Loss Curve\nTraining vs Validation Loss on synthetic dataset")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(epochs, metrics["train"], label="Train AUC")
        plt.plot(epochs, metrics["val"], label="Val AUC")
        plt.title("AUC Curve\nTraining vs Validation AUC on synthetic dataset")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_auc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating AUC curve: {e}")
        plt.close()
