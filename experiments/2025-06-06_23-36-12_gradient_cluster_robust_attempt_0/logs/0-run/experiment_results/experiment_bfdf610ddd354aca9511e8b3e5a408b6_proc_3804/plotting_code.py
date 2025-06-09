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

# Prepare keys
dr_keys = sorted(
    experiment_data.get("dropout_rate", {}).get("synthetic", {}).keys(),
    key=lambda x: float(x),
)

# Plot loss curves
try:
    plt.figure()
    for dr in dr_keys:
        res = experiment_data["dropout_rate"]["synthetic"][dr]
        epochs = range(len(res["losses"]["train"]))
        plt.plot(epochs, res["losses"]["train"], label=f"Train dr={dr}")
        plt.plot(epochs, res["losses"]["val"], label=f"Val dr={dr}")
    plt.title("Loss Curves (Synthetic dataset)\nTrain vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot worst-group accuracy curves
try:
    plt.figure()
    for dr in dr_keys:
        res = experiment_data["dropout_rate"]["synthetic"][dr]
        epochs = range(len(res["metrics"]["train"]))
        plt.plot(epochs, res["metrics"]["train"], label=f"Train dr={dr}")
        plt.plot(epochs, res["metrics"]["val"], label=f"Val dr={dr}")
    plt.title(
        "Worst-group Accuracy Curves (Synthetic dataset)\nTrain vs Validation WG Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Worst-group Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating wg accuracy curves: {e}")
    plt.close()

# Plot test accuracy vs dropout rate
try:
    plt.figure()
    dr_list = [float(dr) for dr in dr_keys]
    acc_list = []
    for dr in dr_keys:
        res = experiment_data["dropout_rate"]["synthetic"][dr]
        preds = np.array(res["predictions"])
        truths = np.array(res["ground_truth"])
        acc = (preds == truths).mean()
        acc_list.append(acc)
        print(f"Dropout {dr}: Test Accuracy = {acc:.4f}")
    plt.bar(dr_list, acc_list)
    plt.title("Test Accuracy vs Dropout Rate (Synthetic dataset)")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Test Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy_vs_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
    plt.close()
