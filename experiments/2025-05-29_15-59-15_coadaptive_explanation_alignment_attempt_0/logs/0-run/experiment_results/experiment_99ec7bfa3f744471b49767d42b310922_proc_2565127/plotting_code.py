import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
# Extract sweep info
sweep = data["learning_rate_sweep"]["user_model"]
lrs = sweep["lrs"]
train_accs = sweep["metrics"]["train"]
val_accs = sweep["metrics"]["val"]
train_losses = sweep["losses"]["train"]
val_losses = sweep["losses"]["val"]
preds = sweep["predictions"]
gts = sweep["ground_truth"]

# Plot train accuracy curves
try:
    plt.figure()
    for lr, acc in zip(lrs, train_accs):
        plt.plot(acc, label=f"lr={lr}")
    plt.title("Training Accuracy (Learning Rate Sweep, User Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "learning_rate_sweep_user_model_train_accuracy.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating train accuracy plot: {e}")
    plt.close()

# Plot validation accuracy curves
try:
    plt.figure()
    for lr, acc in zip(lrs, val_accs):
        plt.plot(acc, label=f"lr={lr}")
    plt.title("Validation Accuracy (Learning Rate Sweep, User Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "learning_rate_sweep_user_model_val_accuracy.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating val accuracy plot: {e}")
    plt.close()

# Plot train loss curves
try:
    plt.figure()
    for lr, loss in zip(lrs, train_losses):
        plt.plot(loss, label=f"lr={lr}")
    plt.title("Training Loss (Learning Rate Sweep, User Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "learning_rate_sweep_user_model_train_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating train loss plot: {e}")
    plt.close()

# Plot validation loss curves
try:
    plt.figure()
    for lr, loss in zip(lrs, val_losses):
        plt.plot(loss, label=f"lr={lr}")
    plt.title("Validation Loss (Learning Rate Sweep, User Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "learning_rate_sweep_user_model_val_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating val loss plot: {e}")
    plt.close()

# Plot test accuracy vs learning rate
try:
    test_acc = [np.mean(np.array(p) == np.array(gt)) for p, gt in zip(preds, gts)]
    plt.figure()
    plt.bar([str(lr) for lr in lrs], test_acc)
    plt.title("Test Accuracy vs Learning Rate (User Model Sweep)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.savefig(
        os.path.join(working_dir, "learning_rate_sweep_user_model_test_accuracy.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
