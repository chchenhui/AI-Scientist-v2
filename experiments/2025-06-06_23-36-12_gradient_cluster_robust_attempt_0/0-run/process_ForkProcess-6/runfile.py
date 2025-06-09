import os
import numpy as np

# Load the saved experiment data
data_path = os.path.join(os.getcwd(), "working", "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# Navigate to the synthetic learning-rate sweep results
synthetic = experiment_data["learning_rate"]["synthetic"]
lrs = synthetic["lrs"]
metrics_train = synthetic["metrics"]["train"]  # shape: (num_lrs, epochs)
metrics_val = synthetic["metrics"]["val"]
losses_train = synthetic["losses"]["train"]
losses_val = synthetic["losses"]["val"]
predictions = synthetic["predictions"]  # shape: (num_lrs, num_test_samples)
ground_truth = synthetic["ground_truth"]

# Extract final-epoch values
final_train_wg = metrics_train[:, -1]
final_val_wg = metrics_val[:, -1]
final_train_loss = losses_train[:, -1]
final_val_loss = losses_val[:, -1]

# Select the best learning rate by highest final validation worst-group accuracy
best_idx = np.argmax(final_val_wg)

# Print metrics with clear labels
print("Training Dataset Metrics:")
print(f"  Final training worst-group accuracy: {final_train_wg[best_idx]:.4f}")
print(f"  Final training average loss: {final_train_loss[best_idx]:.4f}\n")

print("Validation Dataset Metrics:")
print(f"  Final validation worst-group accuracy: {final_val_wg[best_idx]:.4f}")
print(f"  Final validation average loss: {final_val_loss[best_idx]:.4f}\n")

# Compute and print test accuracy
test_acc = np.mean(predictions[best_idx] == ground_truth)
print("Test Dataset Metrics:")
print(f"  Test accuracy: {test_acc:.4f}")
