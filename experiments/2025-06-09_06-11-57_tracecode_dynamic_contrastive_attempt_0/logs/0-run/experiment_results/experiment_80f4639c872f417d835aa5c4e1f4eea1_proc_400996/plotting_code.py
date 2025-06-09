import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
synthetic_data = experiment_data["variable_renaming_invariance"]["synthetic"]

# plot for each epoch configuration
for E in sorted(synthetic_data.keys()):
    data = synthetic_data[E]
    epochs = list(range(1, len(data["losses"]["train"]) + 1))
    try:
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(epochs, data["losses"]["train"], label="Train Loss")
        ax1.plot(epochs, data["losses"]["val"], label="Val Loss")
        ax1.set_title("Left: Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(epochs, data["metrics"]["train"], label="Train Acc")
        ax2.plot(epochs, data["metrics"]["val"], label="Val Acc")
        ax2.plot(epochs, data["metrics"]["rename"], label="Rename Acc")
        ax2.set_title("Right: Training, Validation and Rename Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.suptitle(f"Synthetic Dataset (E={E})")
        save_path = os.path.join(working_dir, f"synthetic_dataset_E{E}_curves.png")
        plt.savefig(save_path)
        plt.close(fig)
        # print final metrics
        last_val = data["metrics"]["val"][-1]
        last_ren = data["metrics"]["rename"][-1]
        print(f"E={E}: final val_acc={last_val:.4f}, rename_acc={last_ren:.4f}")
    except Exception as e:
        print(f"Error creating plot for E={E}: {e}")
        plt.close()
