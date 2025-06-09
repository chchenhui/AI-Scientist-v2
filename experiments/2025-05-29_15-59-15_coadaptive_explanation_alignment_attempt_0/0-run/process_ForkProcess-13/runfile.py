import os
import numpy as np

# Load the saved experiment data
working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# Iterate through each teacher ensemble size and dataset
for size_str, datasets in experiment_data["teacher_ensemble_size"].items():
    for ds_name, ds_data in datasets.items():
        # Extract accuracy metrics
        train_accs = ds_data["metrics"]["train"]
        val_accs = ds_data["metrics"]["val"]
        best_train_acc = float(np.max(train_accs))
        best_val_acc = float(np.max(val_accs))
        # Compute test accuracy
        preds = ds_data["predictions"]
        gt = ds_data["ground_truth"]
        test_acc = float(np.mean(preds == gt))
        # Print dataset and metrics
        print(f"Dataset: {ds_name}")
        print(f"Train accuracy: {best_train_acc:.4f}")
        print(f"Validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}\n")
