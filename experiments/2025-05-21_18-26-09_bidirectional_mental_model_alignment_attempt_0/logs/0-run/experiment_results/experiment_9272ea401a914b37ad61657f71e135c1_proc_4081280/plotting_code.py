import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for ablation, datasets in experiment_data.items():
    for name, data in datasets.items():
        epochs = list(range(1, len(data["losses"]["train"]) + 1))
        # Loss curve
        try:
            plt.figure()
            plt.plot(epochs, data["losses"]["train"], label="Train Loss")
            plt.plot(epochs, data["losses"]["val"], label="Val Loss")
            plt.title(f"{ablation} - {name} Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{name}_{ablation}_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ablation} {name}: {e}")
            plt.close()
        # Accuracy curve
        try:
            plt.figure()
            plt.plot(epochs, data["metrics"]["train"], label="Train Acc")
            plt.plot(epochs, data["metrics"]["val"], label="Val Acc")
            plt.title(f"{ablation} - {name} Accuracy Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{name}_{ablation}_accuracy_curve.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {ablation} {name}: {e}")
            plt.close()
        # Alignment curve
        try:
            plt.figure()
            plt.plot(epochs, data["alignments"]["train"], label="Train Alignment")
            plt.plot(epochs, data["alignments"]["val"], label="Val Alignment")
            plt.title(f"{ablation} - {name} Alignment Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Alignment")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{name}_{ablation}_alignment_curve.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating alignment plot for {ablation} {name}: {e}")
            plt.close()
        # MAI curve
        try:
            plt.figure()
            plt.plot(epochs, data["mai"], label="MAI")
            plt.title(f"{ablation} - {name} MAI Curve")
            plt.xlabel("Epoch")
            plt.ylabel("MAI")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{name}_{ablation}_mai_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating MAI plot for {ablation} {name}: {e}")
            plt.close()
