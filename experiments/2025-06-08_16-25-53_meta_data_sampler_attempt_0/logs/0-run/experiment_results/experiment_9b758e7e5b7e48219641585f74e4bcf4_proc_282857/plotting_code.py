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

for name, exp in experiment_data.get("Ablate_Meta_Inner_Update_Steps", {}).items():
    # Accuracy curves
    try:
        epochs = np.arange(1, len(exp["metrics"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, exp["metrics"]["train"], label="Training")
        plt.plot(epochs, exp["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{name} - Accuracy Curves\nBlue: Training, Orange: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {name}: {e}")
        plt.close()
    # Loss curves
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Training")
        plt.plot(epochs, exp["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} - Loss Curves\nBlue: Training, Orange: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {name}: {e}")
        plt.close()
    # Spearman correlation over meta steps
    try:
        steps = np.arange(1, len(exp["corrs"]) + 1)
        plt.figure()
        plt.plot(steps, exp["corrs"], marker="o")
        plt.xlabel("Meta-Update Step")
        plt.ylabel("Spearman Correlation")
        plt.title(f"{name} - Spearman Correlation over Meta Steps\nDataset: {name}")
        plt.savefig(os.path.join(working_dir, f"{name}_spearman_correlation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Spearman corr plot for {name}: {e}")
        plt.close()
    # N_meta history
    try:
        plt.figure()
        plt.plot(steps, exp["N_meta_history"], marker="o")
        plt.xlabel("Meta-Update Step")
        plt.ylabel("N_meta")
        plt.title(f"{name} - N_meta History\nDataset: {name}")
        plt.savefig(os.path.join(working_dir, f"{name}_N_meta_history.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating N_meta history plot for {name}: {e}")
        plt.close()
    # AUC of validation loss vs token counts
    try:
        plt.figure()
        plt.plot(exp["token_counts"], exp["auc_history"], marker="x")
        plt.xlabel("Token Counts")
        plt.ylabel("AUC of Validation Loss")
        plt.title(f"{name} - AUC of Validation Loss vs Token Counts\nDataset: {name}")
        plt.savefig(os.path.join(working_dir, f"{name}_auc_history.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating AUC history plot for {name}: {e}")
        plt.close()
