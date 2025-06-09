import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# iterate over ablations and plot
for ablation in ["triplet", "contrastive"]:
    data_dict = exp.get(ablation, {}).get("synthetic", {})
    if not data_dict:
        continue

    # Loss curves
    try:
        plt.figure()
        for E, data in data_dict.items():
            epochs = np.arange(1, len(data["losses"]["train"]) + 1)
            plt.plot(epochs, data["losses"]["train"], label=f"train E={E}")
            plt.plot(epochs, data["losses"]["val"], "--", label=f"val E={E}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ablation.capitalize()} Ablation Loss Curves on Synthetic Dataset")
        plt.legend()
        fname = f"{ablation}_loss_curves_synthetic.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {ablation} loss plot: {e}")
        plt.close()

    # Accuracy curves
    try:
        plt.figure()
        for E, data in data_dict.items():
            epochs = np.arange(1, len(data["metrics"]["train"]) + 1)
            plt.plot(epochs, data["metrics"]["train"], label=f"train E={E}")
            plt.plot(epochs, data["metrics"]["val"], "--", label=f"val E={E}")
        plt.xlabel("Epoch")
        plt.ylabel("Retrieval Accuracy")
        plt.title(
            f"{ablation.capitalize()} Ablation Accuracy Curves on Synthetic Dataset"
        )
        plt.legend()
        fname = f"{ablation}_accuracy_curves_synthetic.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {ablation} accuracy plot: {e}")
        plt.close()

    # print final validation accuracies
    for E, data in data_dict.items():
        final_val = data["metrics"]["val"][-1]
        print(f"{ablation} E={E} final_val_acc={final_val:.4f}")
