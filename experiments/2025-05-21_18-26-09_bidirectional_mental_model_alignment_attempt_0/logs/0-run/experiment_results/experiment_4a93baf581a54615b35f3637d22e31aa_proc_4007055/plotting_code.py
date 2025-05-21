import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract synthetic dataset results
exp = experiment_data["dropout_rate"]["synthetic"]
dropout_rates = exp["hyperparams"]
train_losses = exp["losses"]["train"]
val_losses = exp["losses"]["val"]
train_metrics = exp["metrics"]["train"]
val_metrics = exp["metrics"]["val"]
preds_list = exp["predictions"]
gts_list = exp["ground_truth"]

# Plot loss curves
try:
    plt.figure()
    epochs = range(1, len(train_losses[0]) + 1)
    for dr, t_loss, v_loss in zip(dropout_rates, train_losses, val_losses):
        plt.plot(epochs, t_loss, label=f"Train dr={dr}")
        plt.plot(epochs, v_loss, "--", label=f"Val dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Dataset Loss Curves\nTraining (solid) vs Validation (dashed)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot alignment metric curves
try:
    plt.figure()
    for dr, t_met, v_met in zip(dropout_rates, train_metrics, val_metrics):
        plt.plot(epochs, t_met, label=f"Train dr={dr}")
        plt.plot(epochs, v_met, "--", label=f"Val dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment Metric")
    plt.title("Synthetic Dataset Alignment Curves\nAlignment metric (1 - JSD)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment curves plot: {e}")
    plt.close()

# Plot final validation accuracy
try:
    accs = [np.mean(preds == gts) for preds, gts in zip(preds_list, gts_list)]
    plt.figure()
    idx = np.arange(len(dropout_rates))
    plt.bar(idx, accs)
    plt.xticks(idx, dropout_rates)
    plt.xlabel("Dropout Rate")
    plt.ylabel("Accuracy")
    plt.title("Synthetic Dataset Final Validation Accuracy\nAccuracy per Dropout Rate")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy plot: {e}")
    plt.close()

# Print final accuracies
for dr, acc in zip(dropout_rates, accs):
    print(f"Dropout={dr}: Final Val Accuracy={acc:.4f}")
