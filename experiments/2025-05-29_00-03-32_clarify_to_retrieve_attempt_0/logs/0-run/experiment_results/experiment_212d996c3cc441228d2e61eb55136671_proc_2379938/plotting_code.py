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

try:
    data = experiment_data["mc_T_tuning"]["synthetic_xor"]
    plt.figure()
    for mc_T, run in data.items():
        epochs = np.arange(1, len(run["losses"]["train"]) + 1)
        plt.plot(epochs, run["losses"]["train"], label=f"Train Loss T={mc_T}")
        plt.plot(epochs, run["losses"]["val"], "--", label=f"Val Loss T={mc_T}")
    plt.title("Loss Curves\nSynthetic XOR Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    data = experiment_data["mc_T_tuning"]["synthetic_xor"]
    plt.figure()
    for mc_T, run in data.items():
        epochs = np.arange(1, len(run["metrics"]["train"]) + 1)
        plt.plot(epochs, run["metrics"]["train"], label=f"Train CES T={mc_T}")
        plt.plot(epochs, run["metrics"]["val"], "--", label=f"Val CES T={mc_T}")
    plt.title("CES Curves\nSynthetic XOR Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("CES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_CES_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES plot: {e}")
    plt.close()
