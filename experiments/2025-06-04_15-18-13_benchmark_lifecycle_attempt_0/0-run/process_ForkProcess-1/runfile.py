import os
import numpy as np

# 0. Locate the saved experiment file
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

# 1. Load the data dictionary
experiment_data = np.load(file_path, allow_pickle=True).item()

# 2. Extract the recorded metrics
train_losses = experiment_data["original"]["losses"]["train"]
val_losses = experiment_data["original"]["losses"]["val"]
orig_accuracies = experiment_data["original"]["metrics"]["orig_acc"]
aug_accuracies = experiment_data["original"]["metrics"]["aug_acc"]

# 5. Get the final values
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]
final_orig_accuracy = orig_accuracies[-1]
final_aug_accuracy = aug_accuracies[-1]

# 3 & 4. Print each dataset name followed by its metrics with clear labels
print("Training dataset:")
print("final training loss:", f"{final_train_loss:.4f}")

print("Original test dataset:")
print("final validation loss:", f"{final_val_loss:.4f}")
print("final test accuracy:", f"{final_orig_accuracy:.4f}")

print("Augmented test dataset:")
print("final test accuracy:", f"{final_aug_accuracy:.4f}")
