import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

configs = list(experiment_data.keys())
n_epochs = len(experiment_data[configs[0]]["orig"]["losses"]["train"]) if configs else 0

# Plot loss curves
try:
    plt.figure()
    for c in configs:
        train = experiment_data[c]["orig"]["losses"]["train"]
        vo = experiment_data[c]["orig"]["losses"]["val"]
        vr = experiment_data[c]["rot"]["losses"]["val"]
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, label=f"{c} Train")
        plt.plot(epochs, vo, "--", label=f"{c} Orig Val")
        plt.plot(epochs, vr, ":", label=f"{c} Rot Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "MNIST Loss Curves by Augmentation\nSolid: Train, Dashed: Orig Val, Dotted: Rot Val"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_mnist_aug.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure()
    for c in configs:
        ao = experiment_data[c]["orig"]["metrics"]["acc"]
        ar = experiment_data[c]["rot"]["metrics"]["acc"]
        epochs = range(1, len(ao) + 1)
        plt.plot(epochs, ao, label=f"{c} Orig Acc")
        plt.plot(epochs, ar, "--", label=f"{c} Rot Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST Accuracy Curves by Augmentation\nSolid: Orig, Dashed: Rot")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "accuracy_curves_mnist_aug.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot final accuracy comparison
try:
    x = np.arange(len(configs))
    origs = [experiment_data[c]["orig"]["metrics"]["acc"][-1] for c in configs]
    rots = [experiment_data[c]["rot"]["metrics"]["acc"][-1] for c in configs]
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, origs, width, label="Orig Test")
    plt.bar(x + width / 2, rots, width, label="Rot Test")
    plt.xticks(x, configs)
    plt.ylabel("Accuracy")
    plt.title("MNIST Final Accuracy by Augmentation\nBars: Blue=Orig, Orange=Rot")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "final_accuracy_mnist_aug.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy plot: {e}")
    plt.close()

# Print final accuracies
try:
    print("Final test accuracies (Orig, Rot):")
    for c in configs:
        o = experiment_data[c]["orig"]["metrics"]["acc"][-1]
        r = experiment_data[c]["rot"]["metrics"]["acc"][-1]
        print(f"{c}: Orig {o:.4f}, Rot {r:.4f}")
except Exception as e:
    print(f"Error printing final accuracies: {e}")
