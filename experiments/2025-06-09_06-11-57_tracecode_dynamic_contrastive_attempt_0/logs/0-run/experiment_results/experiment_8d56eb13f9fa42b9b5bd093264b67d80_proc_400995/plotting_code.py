import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# load experiment_data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    # Print final validation accuracies
    for dataset_name, results in experiment_data["dead_code_injection"].items():
        for E in sorted(results.keys()):
            val_acc = results[E]["metrics"]["val"][-1]
            print(f"Dataset={dataset_name}, EPOCHS={E}, final val_acc={val_acc:.4f}")

    # Plot loss curves for each dataset
    for dataset_name in experiment_data["dead_code_injection"]:
        try:
            plt.figure()
            data = experiment_data["dead_code_injection"][dataset_name]
            for E in sorted(data.keys()):
                losses = data[E]["losses"]
                epochs = np.arange(1, len(losses["train"]) + 1)
                plt.plot(epochs, losses["train"], label=f"Train E={E}")
                plt.plot(epochs, losses["val"], linestyle="--", label=f"Val E={E}")
            plt.title(f"Loss Curves for {dataset_name} (Train vs Val)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            save_path = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating {dataset_name} loss plot: {e}")
            plt.close()

    # Plot accuracy curves for each dataset
    for dataset_name in experiment_data["dead_code_injection"]:
        try:
            plt.figure()
            data = experiment_data["dead_code_injection"][dataset_name]
            for E in sorted(data.keys()):
                metrics = data[E]["metrics"]
                epochs = np.arange(1, len(metrics["train"]) + 1)
                plt.plot(epochs, metrics["train"], label=f"Train E={E}")
                plt.plot(epochs, metrics["val"], linestyle="--", label=f"Val E={E}")
            plt.title(f"Accuracy Curves for {dataset_name} (Train vs Val)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            save_path = os.path.join(working_dir, f"{dataset_name}_accuracy_curves.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating {dataset_name} accuracy plot: {e}")
            plt.close()
