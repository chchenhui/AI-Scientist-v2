import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot accuracy comparison per dataset
for ds in experiment_data.get("full_DVN", {}):
    try:
        acc_full = experiment_data["full_DVN"][ds]["metrics"]
        acc_lin = experiment_data["linear_DVN"][ds]["metrics"]
        epochs = range(len(acc_full["train"]))

        plt.figure()
        plt.plot(epochs, acc_full["train"], "-o", label="Full DVN Train")
        plt.plot(epochs, acc_full["val"], "-o", label="Full DVN Val")
        plt.plot(epochs, acc_lin["train"], "-s", label="Linear DVN Train")
        plt.plot(epochs, acc_lin["val"], "-s", label="Linear DVN Val")
        plt.title(f"{ds} - Accuracy Comparison\nTraining vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_accuracy_comparison.png")
        plt.savefig(fname)
        plt.close()

        # Print final validation accuracies
        print(f"{ds} full_DVN final val acc: {acc_full['val'][-1]:.4f}")
        print(f"{ds} linear_DVN final val acc: {acc_lin['val'][-1]:.4f}")
    except Exception as e:
        print(f"Error creating plot for {ds}: {e}")
        plt.close()  # Ensure closure on error
