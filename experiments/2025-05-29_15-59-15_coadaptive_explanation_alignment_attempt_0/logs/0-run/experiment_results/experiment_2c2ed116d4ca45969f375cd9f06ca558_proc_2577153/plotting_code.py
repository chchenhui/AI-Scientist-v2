import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
experiment_data = {}
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Prepare for a representative hyperparameter setting
key = "ai_bs_32_user_bs_32"
datasets = list(experiment_data.get("synthetic_diversity", {}).keys())
train_acc_list, val_acc_list = [], []
align_score_list, align_rate_list = [], []
test_acc_list = []

for ds in datasets:
    cfg = experiment_data["synthetic_diversity"][ds]["batch_size"].get(key)
    if cfg is None:
        continue
    m = cfg["metrics"]
    train_acc_list.append(m["train_accs"])
    val_acc_list.append(m["val_accs"])
    align_score_list.append(m["alignment_scores"])
    align_rate_list.append(cfg["alignment_rate"])
    preds, gt = cfg["predictions"], cfg["ground_truth"]
    test_acc_list.append((preds == gt).mean())

# Determine epoch axis
epochs = len(train_acc_list[0]) if train_acc_list else 0
x = np.arange(1, epochs + 1)
dataset_names = datasets[: len(train_acc_list)]

# Plot 1: Training vs Validation Accuracy
try:
    plt.figure()
    for name, ta, va in zip(dataset_names, train_acc_list, val_acc_list):
        plt.plot(x, ta, label=f"{name} Train")
        plt.plot(x, va, label=f"{name} Val")
    plt.title(
        "Training/Validation Accuracy for ai_bs_32_user_bs_32\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "accuracy_ai32_usr32.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 2: Alignment Score Curves
try:
    plt.figure()
    for name, al in zip(dataset_names, align_score_list):
        plt.plot(x, al, label=name)
    plt.title(
        "Alignment Scores over Epochs for ai_bs_32_user_bs_32\nDatasets: "
        + ", ".join(dataset_names)
    )
    plt.xlabel("Epoch")
    plt.ylabel("Alignment Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "alignment_ai32_usr32.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment plot: {e}")
    plt.close()

# Plot 3: Alignment Rates Bar Chart
try:
    plt.figure()
    plt.bar(dataset_names, align_rate_list)
    plt.title(
        "Alignment Rates for ai_bs_32_user_bs_32\nDatasets: " + ", ".join(dataset_names)
    )
    plt.ylabel("Alignment Rate (slope)")
    plt.savefig(os.path.join(working_dir, "alignment_rate_ai32_usr32.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment rate plot: {e}")
    plt.close()

# Plot 4: Test Accuracy Bar Chart
try:
    plt.figure()
    plt.bar(dataset_names, test_acc_list)
    plt.title(
        "Test Accuracy for ai_bs_32_user_bs_32\nDatasets: " + ", ".join(dataset_names)
    )
    plt.ylabel("Test Accuracy")
    plt.savefig(os.path.join(working_dir, "test_accuracy_ai32_usr32.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
