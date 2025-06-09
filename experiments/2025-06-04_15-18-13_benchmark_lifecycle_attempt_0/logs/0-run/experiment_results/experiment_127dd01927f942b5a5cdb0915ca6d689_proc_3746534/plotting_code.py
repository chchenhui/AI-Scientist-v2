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

# Generate plots per model
for model_name, model_data in experiment_data.items():
    # Loss curves
    try:
        plt.figure()
        for eps_key, vals in model_data.items():
            eps = eps_key.replace("eps_", "")
            epochs = range(1, len(vals["losses"]["train"]) + 1)
            plt.plot(epochs, vals["losses"]["train"], label=f"Train ε={eps}")
            plt.plot(epochs, vals["losses"]["val"], "--", label=f"Val ε={eps}")
        plt.title(f"MNIST Training/Validation Loss - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"MNIST_loss_curve_{model_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {model_name}: {e}")
        plt.close()
    # Accuracy curves
    try:
        plt.figure()
        for eps_key, vals in model_data.items():
            eps = eps_key.replace("eps_", "")
            epochs = range(1, len(vals["metrics"]["orig_acc"]) + 1)
            plt.plot(epochs, vals["metrics"]["orig_acc"], label=f"Orig Acc ε={eps}")
            plt.plot(epochs, vals["metrics"]["aug_acc"], "--", label=f"Aug Acc ε={eps}")
        plt.title(f"MNIST Original/Augmented Accuracy - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"MNIST_accuracy_curve_{model_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {model_name}: {e}")
        plt.close()
