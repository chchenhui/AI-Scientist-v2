import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Iterate over each dataset in Ablate_Entropy_Feature
for ds_name, ds_data in experiment_data.get("Ablate_Entropy_Feature", {}).items():
    metrics = ds_data.get("metrics", {})
    train_loss = metrics.get("train_loss", [])
    val_loss = metrics.get("val_loss", [])
    val_acc = metrics.get("val_acc", [])
    corrs = ds_data.get("corrs", [])
    nmeta = ds_data.get("N_meta_history", [])
    epochs = list(range(1, len(train_loss) + 1))

    # Plot training vs validation loss
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"Training and Validation Loss\nLeft: Train Loss, Right: Val Loss | Dataset: {ds_name}"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # Plot validation accuracy
    try:
        plt.figure()
        plt.plot(epochs, val_acc, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Validation Accuracy\nDataset: {ds_name}")
        plt.savefig(os.path.join(working_dir, f"{ds_name}_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # Plot DVN Spearman correlations if available
    if corrs:
        try:
            steps = list(range(1, len(corrs) + 1))
            plt.figure()
            plt.plot(steps, corrs, marker="x")
            plt.xlabel("Meta Update Step")
            plt.ylabel("Spearman Correlation")
            plt.title(f"DVN Correlation vs Meta Updates\nDataset: {ds_name}")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_dvn_correlation.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating correlation plot for {ds_name}: {e}")
            plt.close()

    # Plot N_meta history if available
    if nmeta:
        try:
            steps = list(range(1, len(nmeta) + 1))
            plt.figure()
            plt.plot(steps, nmeta, marker="*")
            plt.xlabel("Meta Update Step")
            plt.ylabel("N_meta")
            plt.title(f"N_meta History\nDataset: {ds_name}")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_n_meta_history.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating N_meta history plot for {ds_name}: {e}")
            plt.close()
