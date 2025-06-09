import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# parse and sort meta-sample sizes
keys = list(experiment_data.keys())
K_keys = []
for k in keys:
    try:
        K_keys.append((int(k.split("=")[1]), k))
    except:
        pass
K_keys.sort(key=lambda x: x[0])
sorted_K_list = [k for k, _ in K_keys]
sorted_keys = [key for _, key in K_keys]

# dataset names
dataset_names = sorted(list(experiment_data[sorted_keys[0]].keys()))

# aggregate final validation metrics vs K
val_acc_vs_K = {ds: [] for ds in dataset_names}
val_loss_vs_K = {ds: [] for ds in dataset_names}
for key in sorted_keys:
    for ds in dataset_names:
        exp = experiment_data[key][ds]
        val_acc_vs_K[ds].append(exp["metrics"]["val"][-1])
        val_loss_vs_K[ds].append(exp["losses"]["val"][-1])

# print evaluation metrics
print("Final Validation Accuracy vs Meta-sample Size:")
for ds, accs in val_acc_vs_K.items():
    print(f"{ds}: {accs}")
print("Final Validation Loss vs Meta-sample Size:")
for ds, losses in val_loss_vs_K.items():
    print(f"{ds}: {losses}")

# choose largest K
K_max = sorted_K_list[-1]
key_max = sorted_keys[-1]

try:
    plt.figure()
    for ds in dataset_names:
        plt.plot(sorted_K_list, val_acc_vs_K[ds], marker="o", label=ds)
    plt.xlabel("Meta-sample Size K")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Final Validation Accuracy vs Meta-sample Size\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, f'final_val_accuracy_vs_K_{"_".join(dataset_names)}.png'
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure()
    for ds in dataset_names:
        plt.plot(sorted_K_list, val_loss_vs_K[ds], marker="o", label=ds)
    plt.xlabel("Meta-sample Size K")
    plt.ylabel("Validation Loss")
    plt.title(
        "Final Validation Loss vs Meta-sample Size\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, f'final_val_loss_vs_K_{"_".join(dataset_names)}.png')
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    plt.figure()
    for ds in dataset_names:
        data = experiment_data[key_max][ds]
        epochs = list(range(1, len(data["metrics"]["train"]) + 1))
        plt.plot(
            epochs,
            data["metrics"]["train"],
            marker="x",
            linestyle="--",
            label=f"{ds} train",
        )
        plt.plot(epochs, data["metrics"]["val"], marker="o", label=f"{ds} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"Epoch Curves at K={K_max}: Training/Validation Accuracy\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir,
            f'epoch_accuracy_curves_K_{K_max}_{"_".join(dataset_names)}.png',
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

try:
    plt.figure()
    for ds in dataset_names:
        data = experiment_data[key_max][ds]
        epochs = list(range(1, len(data["losses"]["train"]) + 1))
        plt.plot(
            epochs,
            data["losses"]["train"],
            marker="x",
            linestyle="--",
            label=f"{ds} train",
        )
        plt.plot(epochs, data["losses"]["val"], marker="o", label=f"{ds} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Epoch Curves at K={K_max}: Training/Validation Loss\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, f'epoch_loss_curves_K_{K_max}_{"_".join(dataset_names)}.png'
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

try:
    plt.figure()
    for ds in dataset_names:
        corrs = experiment_data[key_max][ds]["corrs"]
        plt.plot(range(1, len(corrs) + 1), corrs, marker="o", label=ds)
    plt.xlabel("Meta-update Iteration")
    plt.ylabel("Spearman Correlation")
    plt.title(
        f"K={K_max}: Spearman Correlation Across Meta-updates\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, f'spearman_correlation_K_{K_max}_{"_".join(dataset_names)}.png'
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()
