import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    inits = sorted(exp_data["init_std"].keys(), key=float)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    inits = []

# 1) Loss curves
try:
    plt.figure()
    for std in inits:
        exp = exp_data["init_std"][std]
        epochs = exp["epochs"]
        plt.plot(epochs, exp["losses"]["train"], label=f"train {std}")
        plt.plot(epochs, exp["losses"]["val"], "--", label=f"val {std}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Data Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Alignment curves
try:
    plt.figure()
    for std in inits:
        exp = exp_data["init_std"][std]
        epochs = exp["epochs"]
        plt.plot(epochs, exp["metrics"]["train"], label=f"train align {std}")
        plt.plot(epochs, exp["metrics"]["val"], "--", label=f"val align {std}")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment")
    plt.title("Synthetic Data Alignment Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment curves: {e}")
    plt.close()

# 3) Final validation loss bar chart
try:
    finals = [exp_data["init_std"][s]["losses"]["val"][-1] for s in inits]
    plt.figure()
    plt.bar(inits, finals)
    plt.xlabel("Init Std")
    plt.ylabel("Final Val Loss")
    plt.title("Synthetic Data Final Validation Loss by Init Std")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_loss_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final val loss bar: {e}")
    plt.close()

# 4) Final validation alignment bar chart
try:
    finals_align = [exp_data["init_std"][s]["metrics"]["val"][-1] for s in inits]
    plt.figure()
    plt.bar(inits, finals_align)
    plt.xlabel("Init Std")
    plt.ylabel("Final Val Alignment")
    plt.title("Synthetic Data Final Validation Alignment by Init Std")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_alignment_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final val alignment bar: {e}")
    plt.close()
