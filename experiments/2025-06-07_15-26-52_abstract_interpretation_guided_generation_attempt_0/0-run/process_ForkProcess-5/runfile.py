import os
import numpy as np

# Locate working directory and load experiment data
working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# Iterate over the hyperparameter sweep (learning_rate) and datasets
sweep_data = experiment_data.get("learning_rate", {})
for dataset_name, dataset_data in sweep_data.items():
    params = dataset_data["params"]
    losses = dataset_data["losses"]
    metrics = dataset_data["metrics"]
    # Print final metrics for each learning rate
    for idx, lr in enumerate(params):
        final_train_loss = losses["train"][idx][-1]
        final_val_loss = losses["val"][idx][-1]
        final_train_rate = metrics["train"][idx][-1]
        final_val_rate = metrics["val"][idx][-1]
        print(f"Dataset: {dataset_name} (learning rate = {lr})")
        print(f"Final training loss: {final_train_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Final training generation success rate (AICR): {final_train_rate:.4f}")
        print(
            f"Final validation generation success rate (AICR): {final_val_rate:.4f}\n"
        )
