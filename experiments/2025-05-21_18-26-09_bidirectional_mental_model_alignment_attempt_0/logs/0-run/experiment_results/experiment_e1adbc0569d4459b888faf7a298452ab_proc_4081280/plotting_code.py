import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

if exp:
    datasets = list(next(iter(exp.values())).keys())
    lam_keys = sorted(
        exp.keys(), key=lambda x: float(x[len("lambda_") :].replace("_", "."))
    )

    for name in datasets:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            for lam in lam_keys:
                data = exp[lam][name]
                epochs = range(1, len(data["losses"]["train"]) + 1)
                ax1.plot(epochs, data["losses"]["train"], label=lam)
                ax2.plot(epochs, data["losses"]["val"], label=lam)
            fig.suptitle(f"Loss Curves for {name}")
            ax1.set_title("Left: Training Loss")
            ax2.set_title("Right: Validation Loss")
            ax1.set_xlabel("Epoch")
            ax2.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax2.set_ylabel("Loss")
            ax1.legend()
            ax2.legend()
            plt.savefig(os.path.join(working_dir, f"{name}_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {name}: {e}")
            plt.close()

    for name in datasets:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            for lam in lam_keys:
                data = exp[lam][name]
                epochs = range(1, len(data["metrics"]["train"]) + 1)
                ax1.plot(epochs, data["metrics"]["train"], label=lam)
                ax2.plot(epochs, data["metrics"]["val"], label=lam)
            fig.suptitle(f"Accuracy Curves for {name}")
            ax1.set_title("Left: Training Accuracy")
            ax2.set_title("Right: Validation Accuracy")
            ax1.set_xlabel("Epoch")
            ax2.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax2.set_ylabel("Accuracy")
            ax1.legend()
            ax2.legend()
            plt.savefig(os.path.join(working_dir, f"{name}_accuracy_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {name}: {e}")
            plt.close()

    for name in datasets:
        print(f"Dataset {name}:")
        for lam in lam_keys:
            d = exp[lam][name]
            print(
                f" {lam}: final val_acc={d['metrics']['val'][-1]:.4f}, final MAI={d['mai'][-1]:.4f}"
            )
