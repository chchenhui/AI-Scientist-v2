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

# Prepare averaged metrics across datasets
ab_keys = sorted(
    experiment_data.keys(), key=lambda k: int(k.split("_")[-1])
)  # sort by gamma
loss_avg = {}
align_avg = {}
mai_avg = {}

for ab in ab_keys:
    ds_data = experiment_data[ab]
    # stack per-dataset lists
    train_losses = np.array([ds_data[ds]["losses"]["train"] for ds in ds_data])
    val_losses = np.array([ds_data[ds]["losses"]["val"] for ds in ds_data])
    train_align = np.array([ds_data[ds]["alignments"]["train"] for ds in ds_data])
    val_align = np.array([ds_data[ds]["alignments"]["val"] for ds in ds_data])
    train_mai = np.array([ds_data[ds]["metrics"]["train"] for ds in ds_data])
    val_mai = np.array([ds_data[ds]["metrics"]["val"] for ds in ds_data])
    # average over datasets
    loss_avg[ab] = {"train": train_losses.mean(0), "val": val_losses.mean(0)}
    align_avg[ab] = {"train": train_align.mean(0), "val": val_align.mean(0)}
    mai_avg[ab] = {"train": train_mai.mean(0), "val": val_mai.mean(0)}

# Define epoch axis
if ab_keys:
    epochs = np.arange(1, len(loss_avg[ab_keys[0]]["train"]) + 1)

# Plot average loss
try:
    plt.figure()
    for ab in ab_keys:
        gamma = ab.split("_")[-1]
        plt.plot(epochs, loss_avg[ab]["train"], "-o", label=f"γ={gamma} train")
        plt.plot(epochs, loss_avg[ab]["val"], "--s", label=f"γ={gamma} val")
    plt.title("Avg Loss vs Epoch\nSubtitle: Averaged Across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "avg_loss_gamma_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot average alignment
try:
    plt.figure()
    for ab in ab_keys:
        gamma = ab.split("_")[-1]
        plt.plot(epochs, align_avg[ab]["train"], "-o", label=f"γ={gamma} train")
        plt.plot(epochs, align_avg[ab]["val"], "--s", label=f"γ={gamma} val")
    plt.title("Avg Alignment vs Epoch\nSubtitle: Averaged Across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "avg_alignment_gamma_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment plot: {e}")
    plt.close()

# Plot average MAI
try:
    plt.figure()
    for ab in ab_keys:
        gamma = ab.split("_")[-1]
        plt.plot(epochs, mai_avg[ab]["train"], "-o", label=f"γ={gamma} train")
        plt.plot(epochs, mai_avg[ab]["val"], "--s", label=f"γ={gamma} val")
    plt.title("Avg MAI vs Epoch\nSubtitle: Averaged Across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("MAI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "avg_MAI_gamma_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MAI plot: {e}")
    plt.close()
