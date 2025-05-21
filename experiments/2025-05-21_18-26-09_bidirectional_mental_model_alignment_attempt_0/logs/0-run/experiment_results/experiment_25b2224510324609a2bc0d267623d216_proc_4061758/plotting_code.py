import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
try:
    data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# Per-dataset plots
for ds_name, ds_data in data.items():
    train_losses = ds_data.get("losses", {}).get("train", [])
    val_losses = ds_data.get("losses", {}).get("val", [])
    train_align = ds_data.get("metrics", {}).get("train", [])
    val_align = ds_data.get("metrics", {}).get("val", [])
    epochs = range(1, len(train_losses) + 1)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_losses, marker="o")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].plot(epochs, val_losses, marker="o")
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        fig.suptitle(
            f"{ds_name} Dataset - Loss Curves\nLeft: Training Loss, Right: Validation Loss"
        )
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_align, marker="o")
        axes[0].set_title("Training Alignment")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Alignment (1−JSD)")
        axes[1].plot(epochs, val_align, marker="o")
        axes[1].set_title("Validation Alignment")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Alignment (1−JSD)")
        fig.suptitle(
            f"{ds_name} Dataset - Alignment Curves\nLeft: Training Alignment, Right: Validation Alignment"
        )
        plt.savefig(os.path.join(working_dir, f"{ds_name}_alignment_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating alignment plot for {ds_name}: {e}")
        plt.close()

# Comparison plots across datasets
ds_list = list(data.keys())
if ds_list:
    num_epochs = len(data[ds_list[0]]["losses"]["train"])
    cmp_epochs = range(1, num_epochs + 1)

    try:
        plt.figure()
        for ds in ds_list:
            plt.plot(cmp_epochs, data[ds]["losses"]["train"], marker="o", label=ds)
        plt.title("Comparison of Training Loss Across Datasets")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison training loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        for ds in ds_list:
            plt.plot(cmp_epochs, data[ds]["losses"]["val"], marker="o", label=ds)
        plt.title("Comparison of Validation Loss Across Datasets")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_validation_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison validation loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        for ds in ds_list:
            plt.plot(cmp_epochs, data[ds]["metrics"]["train"], marker="o", label=ds)
        plt.title("Comparison of Training Alignment Across Datasets")
        plt.xlabel("Epoch")
        plt.ylabel("Alignment (1−JSD)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_training_alignment.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison training alignment plot: {e}")
        plt.close()

    try:
        plt.figure()
        for ds in ds_list:
            plt.plot(cmp_epochs, data[ds]["metrics"]["val"], marker="o", label=ds)
        plt.title("Comparison of Validation Alignment Across Datasets")
        plt.xlabel("Epoch")
        plt.ylabel("Alignment (1−JSD)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_validation_alignment.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison validation alignment plot: {e}")
        plt.close()
