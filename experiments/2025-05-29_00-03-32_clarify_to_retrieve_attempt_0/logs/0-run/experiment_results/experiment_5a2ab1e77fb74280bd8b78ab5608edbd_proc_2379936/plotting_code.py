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

weight_decays = [0, 1e-5, 1e-4, 1e-3]

try:
    plt.figure()
    for idx, wd in enumerate(weight_decays):
        train_losses = experiment_data["losses"]["train"][idx]
        val_losses = experiment_data["losses"]["val"][idx]
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f"Train wd={wd}")
        plt.plot(epochs, val_losses, linestyle="--", label=f"Val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (XOR Dataset)\nSolid: Train, Dashed: Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "XOR_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    for idx, wd in enumerate(weight_decays):
        train_ag = experiment_data["accuracy_gain_per_clarification"]["train"][idx]
        val_ag = experiment_data["accuracy_gain_per_clarification"]["val"][idx]
        epochs = np.arange(1, len(train_ag) + 1)
        plt.plot(epochs, train_ag, label=f"Train wd={wd}")
        plt.plot(epochs, val_ag, linestyle="--", label=f"Val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Gain per Clarification")
    plt.title(
        "Accuracy Gain per Clarification (XOR Dataset)\nSolid: Train, Dashed: Validation"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "XOR_accuracy_gain_per_clarification.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy gain plot: {e}")
    plt.close()
