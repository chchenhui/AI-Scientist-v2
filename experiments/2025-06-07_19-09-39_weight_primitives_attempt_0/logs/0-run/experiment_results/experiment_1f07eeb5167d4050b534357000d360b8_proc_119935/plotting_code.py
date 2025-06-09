import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

penalties = ["none", "l1", "l2", "elasticnet"]

try:
    plt.figure()
    for p in penalties:
        tr = data[p]["synthetic"]["metrics"]["train"][0]
        vl = data[p]["synthetic"]["metrics"]["val"][0]
        plt.plot(tr, label=f"{p} train")
        plt.plot(vl, "--", label=f"{p} val")
    plt.suptitle("Dataset: Synthetic")
    plt.title("Training and Validation Error Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves plot: {e}")
    plt.close()

try:
    plt.figure()
    for p in penalties:
        trl = data[p]["synthetic"]["losses"]["train"][0]
        vll = data[p]["synthetic"]["losses"]["val"][0]
        plt.plot(trl, label=f"{p} train")
        plt.plot(vll, "--", label=f"{p} val")
    plt.suptitle("Dataset: Synthetic")
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    finals = [data[p]["synthetic"]["metrics"]["val"][0][-1] for p in penalties]
    x = np.arange(len(penalties))
    plt.figure()
    plt.bar(x, finals, tick_label=penalties)
    plt.suptitle("Dataset: Synthetic")
    plt.title("Final Validation Error per Penalty Type")
    plt.ylabel("Relative Error")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_error.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final val error bar chart: {e}")
    plt.close()
