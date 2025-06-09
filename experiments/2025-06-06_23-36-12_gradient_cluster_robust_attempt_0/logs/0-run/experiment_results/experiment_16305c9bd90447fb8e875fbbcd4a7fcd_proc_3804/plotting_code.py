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

ds = experiment_data.get("momentum_sweep", {}).get("synthetic", {})
moms = ds.get("momentum_values", [])
train_wg = ds.get("metrics", {}).get("train", [])
val_wg = ds.get("metrics", {}).get("val", [])
train_loss = ds.get("losses", {}).get("train", [])
val_loss = ds.get("losses", {}).get("val", [])

epochs = list(range(len(train_wg[0]))) if train_wg else []

# 1. Training worst-group accuracy
try:
    plt.figure()
    for m, wg in zip(moms, train_wg):
        plt.plot(epochs, wg, label=f"momentum={m}")
    plt.title(
        "Synthetic Dataset - Training Worst-Group Accuracy vs Epoch (Momentum Sweep)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Worst-Group Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_train_wg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# 2. Validation worst-group accuracy
try:
    plt.figure()
    for m, wg in zip(moms, val_wg):
        plt.plot(epochs, wg, label=f"momentum={m}")
    plt.title(
        "Synthetic Dataset - Validation Worst-Group Accuracy vs Epoch (Momentum Sweep)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Worst-Group Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_val_wg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# 3. Training loss
try:
    plt.figure()
    for m, ls in zip(moms, train_loss):
        plt.plot(epochs, ls, label=f"momentum={m}")
    plt.title("Synthetic Dataset - Training Loss vs Epoch (Momentum Sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_train_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# 4. Validation loss
try:
    plt.figure()
    for m, ls in zip(moms, val_loss):
        plt.plot(epochs, ls, label=f"momentum={m}")
    plt.title("Synthetic Dataset - Validation Loss vs Epoch (Momentum Sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()
