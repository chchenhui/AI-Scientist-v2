import os
import numpy as np

# Locate and load the saved experiment data
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()

# Extract the synthetic dataset under classification_head_depth
synthetic = experiment_data["classification_head_depth"]["synthetic"]
head_depths = synthetic["head_depths"]
train_losses = synthetic["losses"]["train"]
val_losses = synthetic["losses"]["val"]
train_accs = synthetic["classification_accuracy"]["train"]
val_accs = synthetic["classification_accuracy"]["val"]
train_rates = synthetic["metrics"]["train"]
val_rates = synthetic["metrics"]["val"]

# Print the final metrics
print("Dataset: synthetic")
for idx, depth in enumerate(head_depths):
    print(f"\nHead depth {depth}:")
    print(f"train loss: {train_losses[idx][-1]:.4f}")
    print(f"validation loss: {val_losses[idx][-1]:.4f}")
    print(f"train accuracy: {train_accs[idx][-1]:.4f}")
    print(f"validation accuracy: {val_accs[idx][-1]:.4f}")
    print(f"train generation pass rate: {train_rates[idx][-1]:.4f}")
    print(f"validation generation pass rate: {val_rates[idx][-1]:.4f}")
