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
    data = {}

# Plot loss curves
try:
    plt.figure()
    wds = sorted(data.get("weight_decay", {}).keys(), key=lambda x: float(x))
    epochs = range(len(next(iter(data["weight_decay"].values()))["losses"]["train"]))
    for wd in wds:
        losses = data["weight_decay"][wd]["losses"]
        plt.plot(epochs, losses["train"], label=f"train wd={wd}")
        plt.plot(epochs, losses["val"], "--", label=f"val wd={wd}")
    plt.title("Loss Curves (RandomSeqDataset)\nTrain (solid) vs Val (dashed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_randomseqdataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot memory retention ratio curves
try:
    plt.figure()
    for wd in wds:
        mets = data["weight_decay"][wd]["metrics"]
        plt.plot(epochs, mets["train"], label=f"train wd={wd}")
        plt.plot(epochs, mets["val"], "--", label=f"val wd={wd}")
    plt.title(
        "Memory Retention Ratio (RandomSeqDataset)\nTrain (solid) vs Val (dashed)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ratio_curves_randomseqdataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ratio curves: {e}")
    plt.close()

# Plot best ground truth vs generated
try:
    # find best wd by final val loss
    best_wd = min(wds, key=lambda wd: data["weight_decay"][wd]["losses"]["val"][-1])
    rec = data["weight_decay"][best_wd]
    gt = np.array(rec["ground_truth"])
    preds = np.array(rec["predictions"])
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(gt, "-o")
    axs[0].set_title("Ground Truth")
    axs[1].plot(preds, "-o")
    axs[1].set_title("Generated Samples")
    fig.suptitle(
        "Generated Samples vs Ground Truth (RandomSeqDataset)\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.savefig(os.path.join(working_dir, "best_wd_sequence_randomseqdataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating prediction plot: {e}")
    plt.close()

# Print best metric
if data:
    final_loss = data["weight_decay"][best_wd]["losses"]["val"][-1]
    print(f"Best weight_decay: {best_wd}, final validation loss: {final_loss:.4f}")
