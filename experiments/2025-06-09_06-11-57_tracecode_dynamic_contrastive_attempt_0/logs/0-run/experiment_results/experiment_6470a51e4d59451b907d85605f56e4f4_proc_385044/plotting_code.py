import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# load data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    syn = data["embed_dim"]["synthetic"]
    dims = syn["values"]
    acc_train = syn["metrics"]["train"]
    acc_val = syn["metrics"]["val"]
    loss_train = syn["losses"]["train"]
    loss_val = syn["losses"]["val"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise

# Plot accuracy vs epochs
try:
    plt.figure()
    for d, tr, vl in zip(dims, acc_train, acc_val):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{d} Train")
        plt.plot(epochs, vl, linestyle="--", label=f"{d} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Retrieval Accuracy")
    plt.title("Synthetic: Retrieval Accuracy vs Epochs\nTrain (solid) and Val (dashed)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_vs_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot loss vs epochs
try:
    plt.figure()
    for d, tr, vl in zip(dims, loss_train, loss_val):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{d} Train")
        plt.plot(epochs, vl, linestyle="--", label=f"{d} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")
    plt.title(
        "Synthetic: Training and Validation Loss vs Epochs\nTrain (solid) and Val (dashed)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_vs_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot final validation accuracy vs embed_dim
try:
    final_acc = [v[-1] for v in acc_val]
    plt.figure()
    plt.bar(dims, final_acc, color="skyblue")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Final Validation Accuracy")
    plt.title("Synthetic: Final Validation Accuracy vs Embedding Dimension")
    plt.xticks(dims)
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_acc_vs_embed_dim.png"))
    plt.close()
except Exception as e:
    print(f"Error creating summary bar chart: {e}")
    plt.close()
