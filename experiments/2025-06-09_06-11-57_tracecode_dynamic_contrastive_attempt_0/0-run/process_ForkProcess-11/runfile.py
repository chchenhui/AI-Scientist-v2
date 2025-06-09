import os
import numpy as np

# Load the experiment data
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()

# Iterate over each dataset under the "EPOCHS" key
for dataset_name, epoch_dict in experiment_data.get("EPOCHS", {}).items():
    print(f"Dataset: {dataset_name}")
    # Sort by epoch count for consistent ordering
    for E in sorted(epoch_dict):
        data = epoch_dict[E]
        train_acc = data["metrics"]["train"][-1]
        val_acc = data["metrics"]["val"][-1]
        train_loss = data["losses"]["train"][-1]
        val_loss = data["losses"]["val"][-1]
        print(f"EPOCHS = {E}")
        print(f"train accuracy: {train_acc:.4f}")
        print(f"validation accuracy: {val_acc:.4f}")
        print(f"training loss: {train_loss:.4f}")
        print(f"validation loss: {val_loss:.4f}")
        print()
