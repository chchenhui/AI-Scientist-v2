import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["bidirectional_lstm_ablation"]["synthetic"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Loss curves plot
try:
    plt.figure()
    for variant, results in exp.items():
        for E, data in results.items():
            tr = data["losses"]["train"]
            vl = data["losses"]["val"]
            plt.plot(tr, label=f"{variant} train E={E}")
            plt.plot(vl, "--", label=f"{variant} val E={E}")
    plt.title(
        "Training vs Validation Loss\nsolid: train, dashed: validation on synthetic dataset"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_bidirectional_lstm_ablation_loss_curves.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Accuracy curves plot
try:
    plt.figure()
    for variant, results in exp.items():
        for E, data in results.items():
            ta = data["metrics"]["train"]
            va = data["metrics"]["val"]
            plt.plot(ta, label=f"{variant} train E={E}")
            plt.plot(va, "--", label=f"{variant} val E={E}")
    plt.title(
        "Retrieval Accuracy vs Epoch\nsolid: train, dashed: validation on synthetic dataset"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_bidirectional_lstm_ablation_accuracy_curves.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
