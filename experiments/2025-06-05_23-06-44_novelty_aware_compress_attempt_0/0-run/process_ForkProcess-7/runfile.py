import os
import numpy as np

# locate and load the saved experiment data
working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# iterate through each learning rate and dataset
for lr, lr_experiments in experiment_data["learning_rate"].items():
    for dataset_name, results in lr_experiments.items():
        train_ratios = results["metrics"]["train"]
        val_ratios = results["metrics"]["val"]
        final_train_ratio = train_ratios[-1]
        final_val_ratio = val_ratios[-1]
        # print dataset and metric values
        print(f"Dataset: {dataset_name} (learning rate: {lr})")
        print(f"  train memory retention ratio: {final_train_ratio:.4f}")
        print(f"  validation memory retention ratio: {final_val_ratio:.4f}")
