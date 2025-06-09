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

# Plot loss curves
try:
    plt.figure()
    data = experiment_data["lstm_bidirectional"]
    for key, val in data.items():
        losses = val["synthetic"]["losses"]
        plt.plot(losses["train"], label=f"train bidirectional={key}")
        plt.plot(losses["val"], label=f"val bidirectional={key}")
    plt.title("Training & Validation Loss Curves\nSynthetic dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "lstm_bidirectional_synthetic_loss_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure()
    for key, val in data.items():
        metrics = val["synthetic"]["metrics"]
        plt.plot(metrics["train"], label=f"train bidirectional={key}")
        plt.plot(metrics["val"], label=f"val bidirectional={key}")
    plt.title("Training & Validation Retrieval Accuracy\nSynthetic dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "lstm_bidirectional_synthetic_accuracy_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# Plot final validation accuracy bar chart
try:
    keys = list(data.keys())
    final_accs = [data[k]["synthetic"]["metrics"]["val"][-1] for k in keys]
    plt.figure()
    plt.bar([f"bidirectional={k}" for k in keys], final_accs, color=["C0", "C1"])
    plt.title("Final Validation Retrieval Accuracy\nSynthetic dataset")
    plt.xlabel("Model Setting")
    plt.ylabel("Accuracy")
    plt.savefig(
        os.path.join(working_dir, "lstm_bidirectional_synthetic_final_val_accuracy.png")
    )
    plt.close()
    print("Final validation accuracies:", dict(zip(keys, final_accs)))
except Exception as e:
    print(f"Error creating final val accuracy bar chart: {e}")
    plt.close()
