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

models = ["MLP", "CNN"]
bs_keys = sorted(experiment_data.get("batch_size_sweep", {}).keys(), key=int)

for model_name in models:
    try:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for bs in bs_keys:
            mdata = experiment_data["batch_size_sweep"][bs][model_name]
            epochs = range(1, len(mdata["losses"]["train"]) + 1)
            axs[0].plot(epochs, mdata["losses"]["train"], label=f"BS {bs}")
            axs[1].plot(epochs, mdata["losses"]["val"], label=f"BS {bs}")
        axs[0].set_title("Left: Training Loss, Dataset: MNIST")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[1].set_title("Right: Validation Loss, Dataset: MNIST")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        fig.suptitle(f"MNIST Loss Curves - {model_name}")
        plt.savefig(
            os.path.join(working_dir, f"mnist_{model_name.lower()}_loss_curves.png")
        )
        plt.close(fig)
    except Exception as e:
        print(f"Error creating {model_name} loss plot: {e}")
        plt.close()

print("\nFinal Test Accuracies (Original / Augmented):")
for bs in bs_keys:
    for model_name in models:
        mdata = experiment_data["batch_size_sweep"][bs][model_name]
        oacc = mdata["metrics"]["orig_acc"][-1]
        aacc = mdata["metrics"]["aug_acc"][-1]
        print(f"BS {bs} {model_name}: orig_acc={oacc:.4f}, aug_acc={aacc:.4f}")
