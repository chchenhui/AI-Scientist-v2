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

# Label smoothing summary
try:
    ls = experiment_data["label_smoothing"]
    eps_items = sorted((float(k.split("_")[1]), k) for k in ls.keys())
    epochs = range(1, len(next(iter(ls.values()))["losses"]["train"]) + 1)
    plt.figure(figsize=(10, 4))
    # Loss curves
    plt.subplot(1, 2, 1)
    for eps, key in eps_items:
        tr = ls[key]["losses"]["train"]
        vl = ls[key]["losses"]["val"]
        plt.plot(epochs, tr, label=f"train ε={eps}")
        plt.plot(epochs, vl, "--", label=f"val ε={eps}")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Accuracy curves
    plt.subplot(1, 2, 2)
    for eps, key in eps_items:
        orig = ls[key]["metrics"]["orig_acc"]
        aug = ls[key]["metrics"]["aug_acc"]
        plt.plot(epochs, orig, label=f"orig ε={eps}")
        plt.plot(epochs, aug, "--", label=f"aug ε={eps}")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.suptitle("Label Smoothing on MNIST\nLeft: Loss Curves, Right: Accuracy Curves")
    plt.savefig(os.path.join(working_dir, "mnist_label_smoothing_summary.png"))
    plt.close()
except Exception as e:
    print(f"Error creating label smoothing summary plot: {e}")
    plt.close()

# Adversarial training loss curves
try:
    adv = experiment_data["adversarial_training"]
    eps_items = sorted((float(k.split("_")[1]), k) for k in adv.keys())
    epochs = range(1, len(next(iter(adv.values()))["clean"]["losses"]["train"]) + 1)
    plt.figure(figsize=(10, 4))
    # Clean training loss
    plt.subplot(1, 2, 1)
    for eps, key in eps_items:
        tr = adv[key]["clean"]["losses"]["train"]
        vl = adv[key]["clean"]["losses"]["val"]
        plt.plot(epochs, tr, label=f"clean train ε={eps}")
        plt.plot(epochs, vl, "--", label=f"clean val ε={eps}")
    plt.title("Clean Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Adversarial training loss
    plt.subplot(1, 2, 2)
    for eps, key in eps_items:
        tr = adv[key]["adv"]["losses"]["train"]
        vl = adv[key]["adv"]["losses"]["val"]
        plt.plot(epochs, tr, label=f"adv train ε={eps}")
        plt.plot(epochs, vl, "--", label=f"adv val ε={eps}")
    plt.title("Adversarial Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.suptitle(
        "Adversarial Training on MNIST\nLeft: Clean, Right: Adversarial Loss Curves"
    )
    plt.savefig(os.path.join(working_dir, "mnist_adversarial_training_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating adversarial training loss plot: {e}")
    plt.close()

# Adversarial training accuracy curves
try:
    plt.figure(figsize=(10, 4))
    # Original accuracy
    plt.subplot(1, 2, 1)
    for eps, key in eps_items:
        oc = adv[key]["clean"]["metrics"]["orig_acc"]
        oa = adv[key]["adv"]["metrics"]["orig_acc"]
        plt.plot(epochs, oc, label=f"clean ε={eps}")
        plt.plot(epochs, oa, "--", label=f"adv ε={eps}")
    plt.title("Original Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # Robust accuracy
    plt.subplot(1, 2, 2)
    for eps, key in eps_items:
        rc = adv[key]["clean"]["metrics"]["robust_acc"]
        ra = adv[key]["adv"]["metrics"]["robust_acc"]
        plt.plot(epochs, rc, label=f"clean ε={eps}")
        plt.plot(epochs, ra, "--", label=f"adv ε={eps}")
    plt.title("Robust Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.suptitle(
        "Adversarial Training on MNIST\nLeft: Original, Right: Robust Accuracy"
    )
    plt.savefig(os.path.join(working_dir, "mnist_adversarial_training_acc_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating adversarial training accuracy plot: {e}")
    plt.close()

# Print final metrics
try:
    print(f"Label Smoothing Final Metrics (Epoch {len(epochs)}):")
    for eps, key in eps_items:
        m = ls[key]["metrics"]
        print(
            f"ε={eps}: orig_acc={m['orig_acc'][-1]:.4f}, aug_acc={m['aug_acc'][-1]:.4f}"
        )
    print(f"Adversarial Training Final Metrics (Epoch {len(epochs)}):")
    for eps, key in eps_items:
        mc = adv[key]["clean"]["metrics"]
        ma = adv[key]["adv"]["metrics"]
        print(
            f"ε={eps}, clean: orig={mc['orig_acc'][-1]:.4f}, robust={mc['robust_acc'][-1]:.4f}"
        )
        print(
            f"ε={eps}, adv:   orig={ma['orig_acc'][-1]:.4f}, robust={ma['robust_acc'][-1]:.4f}"
        )
except Exception as e:
    print(f"Error printing final metrics: {e}")
