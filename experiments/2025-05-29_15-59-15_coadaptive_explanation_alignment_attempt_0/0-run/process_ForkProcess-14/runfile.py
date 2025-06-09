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

# Flatten all dataset configurations into a list
items = []
for size, cfgs in experiment_data.get("teacher_ensemble_size", {}).items():
    for ds_name, data in cfgs.items():
        items.append((size, ds_name, data))

# Sample at most five configurations evenly
max_plots = 5
total = len(items)
step = max(1, total // max_plots)
selected = [items[i] for i in range(0, total, step)][:max_plots]

# Plot training vs validation accuracy for each selected config
for idx, (_size, ds_name, data) in enumerate(selected, 1):
    try:
        plt.figure()
        epochs = np.arange(1, len(data["metrics"]["train"]) + 1)
        plt.plot(epochs, data["metrics"]["train"], label="Train Acc")
        plt.plot(epochs, data["metrics"]["val"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curves for {ds_name}")
        plt.suptitle("Training vs Validation Accuracy", fontsize=10)
        plt.legend()
        fname = f"{ds_name}_accuracy_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot{idx}: {e}")
        plt.close()
