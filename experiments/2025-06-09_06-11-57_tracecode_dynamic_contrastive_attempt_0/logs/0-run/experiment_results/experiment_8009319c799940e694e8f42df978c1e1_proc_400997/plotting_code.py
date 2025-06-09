import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["embed_hidden"]["synthetic"]
    dims = sorted(data.keys())
    epochs = len(data[dims[0]]["losses"]["train"])
    xs = list(range(1, epochs + 1))
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data, dims, xs = {}, [], []

# Loss curves
try:
    plt.figure()
    for d in dims:
        l = data[d]["losses"]
        plt.plot(xs, l["train"], label=f"train dim={d}")
        plt.plot(xs, l["val"], "--", label=f"val dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves on synthetic dataset\nTrain: solid, Val: dashed")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# Accuracy curves
try:
    plt.figure()
    for d in dims:
        m = data[d]["metrics"]
        plt.plot(xs, m["train"], label=f"train dim={d}")
        plt.plot(xs, m["val"], "--", label=f"val dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Retrieval Accuracy")
    plt.title("Accuracy Curves on synthetic dataset\nTrain: solid, Val: dashed")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# Print final validation accuracies
for d in dims:
    final_acc = data[d]["metrics"]["val"][-1]
    print(f"Final val acc for dim={d}: {final_acc:.4f}")
