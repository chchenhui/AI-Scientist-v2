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

# Group data by dataset name
group_data = {}
for size, cfgs in experiment_data.get("teacher_ensemble_size", {}).items():
    for ds_name, data in cfgs.items():
        group_data.setdefault(ds_name, []).append(data)

# Select at most five datasets evenly
ds_names = list(group_data.keys())
max_plots = 5
step = max(1, len(ds_names) // max_plots)
selected_ds = ds_names[0::step][:max_plots]

for ds_name in selected_ds:
    # Collect train and validation curves
    train_list, val_list = [], []
    for d in group_data[ds_name]:
        m = d.get("metrics", {})
        if "train" in m:
            train_list.append(np.array(m["train"]))
        if "val" in m:
            val_list.append(np.array(m["val"]))
    if not train_list:
        continue

    # Compute mean and SE for train
    min_train = min(arr.shape[0] for arr in train_list)
    train_arr = np.stack([arr[:min_train] for arr in train_list], axis=0)
    train_mean = np.mean(train_arr, axis=0)
    train_se = np.std(train_arr, axis=0) / np.sqrt(train_arr.shape[0])

    # Compute mean and SE for validation (if available)
    val_mean = val_se = None
    if val_list:
        min_val = min(arr.shape[0] for arr in val_list)
        val_arr = np.stack([arr[:min_val] for arr in val_list], axis=0)
        val_mean = np.mean(val_arr, axis=0)
        val_se = np.std(val_arr, axis=0) / np.sqrt(val_arr.shape[0])

    # Print final metrics
    print(f"{ds_name} final Train Acc = {train_mean[-1]:.4f} ± {train_se[-1]:.4f}")
    if val_mean is not None:
        print(f"{ds_name} final Val   Acc = {val_mean[-1]:.4f} ± {val_se[-1]:.4f}")

    # Plot with mean and SE
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        epochs = np.arange(1, min_train + 1)
        axes[0].plot(epochs, train_mean, label="Mean")
        axes[0].fill_between(
            epochs, train_mean - train_se, train_mean + train_se, alpha=0.3, label="SE"
        )
        axes[0].set_title("Train Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        if val_mean is not None:
            epochs_val = np.arange(1, val_mean.shape[0] + 1)
            axes[1].plot(epochs_val, val_mean, label="Mean")
            axes[1].fill_between(
                epochs_val, val_mean - val_se, val_mean + val_se, alpha=0.3, label="SE"
            )
            axes[1].set_title("Validation Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()
        else:
            axes[1].axis("off")

        fig.suptitle(
            f"{ds_name}: Accuracy with Mean and SE (Left: Train, Right: Validation)",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"{ds_name}_mean_se_accuracy.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating mean/se plot for {ds_name}: {e}")
        plt.close()
