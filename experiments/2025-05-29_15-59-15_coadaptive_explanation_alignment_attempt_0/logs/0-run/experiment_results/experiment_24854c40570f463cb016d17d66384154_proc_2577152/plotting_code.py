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

key = "ai_bs_32_user_bs_32"
scenarios = [
    "CE_hard_labels",
    "soft_label_distillation",
    "bias_awareness",
    "dual_channel",
]

for scenario in scenarios:
    try:
        metrics = experiment_data.get(scenario, {}).get(key, {}).get("metrics", {})
        train_acc = metrics.get("train")
        val_acc = metrics.get("val")
        if train_acc is None or val_acc is None:
            raise ValueError("Missing accuracy data")
        epochs = np.arange(1, len(train_acc) + 1)
        plt.figure()
        plt.plot(epochs, train_acc, label="Training Acc", color="blue")
        plt.plot(epochs, val_acc, label="Validation Acc", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(
            f"{scenario}: Training vs Validation Accuracy\n"
            "Training=Blue, Validation=Orange | Dataset: Synthetic Binary"
        )
        fname = f"{scenario}_accuracy_ai32_usr32.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {scenario} accuracy plot: {e}")
        plt.close()

try:
    align = experiment_data["dual_channel"][key]["metrics"]["alignment_rate"]
    epochs = np.arange(1, len(align) + 1)
    plt.figure()
    plt.plot(epochs, align, label="Alignment Rate", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment Rate")
    plt.legend()
    plt.title("dual_channel: Alignment Rate Over Epochs\nDataset: Synthetic Binary")
    plt.savefig(os.path.join(working_dir, "dual_channel_alignment_ai32_usr32.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dual_channel alignment plot: {e}")
    plt.close()
