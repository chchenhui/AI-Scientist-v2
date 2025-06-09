import os
import numpy as np

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# Iterate over each dataset in the experiment data
for dataset_name, dataset_content in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    # Extract final loss values
    train_loss_list = dataset_content["losses"]["train"]
    val_loss_list = dataset_content["losses"]["val"]
    final_train_loss = train_loss_list[-1] if train_loss_list else None
    final_val_loss = val_loss_list[-1] if val_loss_list else None

    # Extract final error-free generation rates
    train_error_rate_list = dataset_content["metrics"]["train"]
    val_error_rate_list = dataset_content["metrics"]["val"]
    final_train_error_rate = (
        train_error_rate_list[-1] if train_error_rate_list else None
    )
    final_val_error_rate = val_error_rate_list[-1] if val_error_rate_list else None

    # Print metrics with clear labels
    if final_train_loss is not None:
        print(f"Final training loss: {final_train_loss:.4f}")
    if final_val_loss is not None:
        print(f"Final validation loss: {final_val_loss:.4f}")
    if final_train_error_rate is not None:
        print(
            f"Final training error-free generation rate: {final_train_error_rate:.4f}"
        )
    if final_val_error_rate is not None:
        print(
            f"Final validation error-free generation rate: {final_val_error_rate:.4f}"
        )
    print()
