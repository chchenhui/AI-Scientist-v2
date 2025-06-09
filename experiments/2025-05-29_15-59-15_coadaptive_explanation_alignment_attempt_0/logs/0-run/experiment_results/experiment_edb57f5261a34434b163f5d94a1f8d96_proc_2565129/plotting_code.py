import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")

# Load experiment data
try:
    exp = np.load(data_path, allow_pickle=True).item()
    sd = exp["weight_decay"]["static_explainer"]
    wds = np.array(sd["weight_decays"])
    train_accs = sd["metrics"]["train"]
    val_accs = sd["metrics"]["val"]
    train_losses = sd["losses"]["train"]
    val_losses = sd["losses"]["val"]
    preds = sd["predictions"]
    gt = np.array(sd["ground_truth"])
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sd = None

# Plot accuracy curves
try:
    if sd:
        plt.figure()
        epochs = range(1, len(train_accs[0]) + 1)
        for i, wd in enumerate(wds):
            plt.plot(epochs, train_accs[i], "--", label=f"Train wd={wd:.1e}")
            plt.plot(epochs, val_accs[i], "-", label=f"Val wd={wd:.1e}")
        plt.title("Accuracy Curves (Static Explainer, Weight Decay Sweep)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "static_explainer_weight_decay_accuracy.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot loss curves
try:
    if sd:
        plt.figure()
        for i, wd in enumerate(wds):
            plt.plot(epochs, train_losses[i], "--", label=f"Train wd={wd:.1e}")
            plt.plot(epochs, val_losses[i], "-", label=f"Val wd={wd:.1e}")
        plt.title("Loss Curves (Static Explainer, Weight Decay Sweep)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "static_explainer_weight_decay_loss.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot test accuracy vs weight decay
try:
    if sd:
        test_acc = [np.mean(np.array(p) == gt) for p in preds]
        plt.figure()
        plt.bar([f"{wd:.1e}" for wd in wds], test_acc)
        plt.title("Test Accuracy vs Weight Decay (Static Explainer)")
        plt.xlabel("Weight Decay")
        plt.ylabel("Test Accuracy")
        plt.savefig(
            os.path.join(working_dir, "static_explainer_weight_decay_test_accuracy.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
