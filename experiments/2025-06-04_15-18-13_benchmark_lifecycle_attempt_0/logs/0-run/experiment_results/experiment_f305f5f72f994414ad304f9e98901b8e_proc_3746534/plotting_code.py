import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot loss and accuracy curves per dataset
for ds_name, ds_dict in experiment_data.get("multi_dataset_generalization", {}).items():
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for style, eps_key in zip(
            ["-", "--"], sorted(ds_dict.keys(), key=lambda x: float(x.split("_")[1]))
        ):
            data = ds_dict[eps_key]
            epochs = np.arange(1, len(data["losses"]["train"]) + 1)
            axes[0].plot(
                epochs, data["losses"]["train"], style, label=f"Train {eps_key}"
            )
            axes[0].plot(
                epochs, data["losses"]["val"], style, label=f"Val {eps_key}", alpha=0.7
            )
            axes[1].plot(
                epochs, data["metrics"]["train"], style, label=f"Train {eps_key}"
            )
            axes[1].plot(
                epochs, data["metrics"]["val"], style, label=f"Val {eps_key}", alpha=0.7
            )
        axes[0].set_title("Left: Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[1].set_title("Right: Accuracy Curves")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        fig.suptitle(f"{ds_name}: Loss & Accuracy Curves")
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_accuracy_curves.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating {ds_name} loss & accuracy plot: {e}")
        plt.close("all")

# Print final validation accuracy for each setting
for ds_name, ds_dict in experiment_data.get("multi_dataset_generalization", {}).items():
    for eps_key, data in ds_dict.items():
        final_acc = data["metrics"]["val"][-1]
        print(f"{ds_name} {eps_key} final val accuracy: {final_acc:.4f}")
