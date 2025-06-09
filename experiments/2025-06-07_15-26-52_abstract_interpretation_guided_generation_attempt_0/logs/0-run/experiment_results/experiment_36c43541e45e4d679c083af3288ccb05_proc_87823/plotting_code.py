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

data = experiment_data.get("embedding_dim", {}).get("synthetic", {})
params = data.get("params", [])
loss_train = data.get("losses", {}).get("train", [])
loss_val = data.get("losses", {}).get("val", [])
metric_train = data.get("metrics", {}).get("train", [])
metric_val = data.get("metrics", {}).get("val", [])
epochs = range(1, len(loss_train[0]) + 1) if loss_train else []

try:
    plt.figure()
    for i, emb in enumerate(params):
        plt.plot(epochs, loss_train[i], label=f"train dim={emb}")
        plt.plot(epochs, loss_val[i], "--", label=f"val dim={emb}")
    plt.title("Loss curves on synthetic dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    for i, emb in enumerate(params):
        plt.plot(epochs, metric_train[i], label=f"train dim={emb}")
        plt.plot(epochs, metric_val[i], "--", label=f"val dim={emb}")
    plt.title("Generation accuracy rates on synthetic dataset")
    plt.xlabel("Epoch")
    plt.ylabel("AICR Rate")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_accuracy_rates.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy rates plot: {e}")
    plt.close()
