import os
import numpy as np

# Load saved experiment data
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()

# Iterate through each experiment configuration
for n_epochs, run_data in experiment_data["n_epochs"].items():
    print(f"n_epochs: {n_epochs}")
    # Iterate through each model in the run
    for model_name, model_data in run_data["models"].items():
        print(f"Model: {model_name}")
        # Extract final epoch metrics
        training_loss = model_data["losses"]["train"][-1]
        validation_loss = model_data["losses"]["val"][-1]
        original_test_accuracy = model_data["metrics"]["orig_acc"][-1]
        augmented_test_accuracy = model_data["metrics"]["aug_acc"][-1]
        # Print metrics by dataset
        print("Dataset: Training")
        print(f"training loss: {training_loss:.4f}")
        print("Dataset: Validation")
        print(f"validation loss: {validation_loss:.4f}")
        print("Dataset: Original Test")
        print(f"original test accuracy: {original_test_accuracy:.4f}")
        print("Dataset: Augmented Test")
        print(f"augmented test accuracy: {augmented_test_accuracy:.4f}")
        print()
