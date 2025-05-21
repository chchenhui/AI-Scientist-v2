import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    synth = experiment_data["weight_decay"]["synthetic"]
    wds = synth["weight_decay_values"]
    tr_losses = synth["losses"]["train"]
    vl_losses = synth["losses"]["val"]
    tr_align = synth["metrics"]["train"]
    vl_align = synth["metrics"]["val"]
    preds = synth["predictions"]
    gt = synth["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    tr_losses = vl_losses = tr_align = vl_align = preds = gt = wds = None

# Print final validation metrics
if wds is not None:
    final_vl = vl_losses[:, -1]
    final_va = vl_align[:, -1]
    print("Weight decays:", wds)
    print("Final val loss:", final_vl)
    print("Final val alignment:", final_va)

# Plot loss curves
try:
    plt.figure()
    epochs = np.arange(1, tr_losses.shape[1] + 1)
    for i, wd in enumerate(wds):
        plt.plot(epochs, tr_losses[i], label=f"train wd={wd}")
        plt.plot(epochs, vl_losses[i], "--", label=f"val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves - Synthetic dataset\nSolid: train, Dashed: val")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot alignment curves
try:
    plt.figure()
    for i, wd in enumerate(wds):
        plt.plot(epochs, tr_align[i], label=f"train wd={wd}")
        plt.plot(epochs, vl_align[i], "--", label=f"val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment")
    plt.title("Alignment Curves - Synthetic dataset\nSolid: train, Dashed: val")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment curves: {e}")
    plt.close()

# Plot predictions vs ground truth for best model
if preds is not None:
    best_idx = int(np.argmax(vl_align[:, -1]))
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(gt, "o", label="Ground Truth")
        plt.plot(preds[best_idx], "^", label="Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Class")
        plt.title(
            f"Prediction vs Ground Truth - Synthetic dataset (wd={wds[best_idx]})\nSymbols: o=GT, ^=Pred"
        )
        plt.legend()
        fname = f"synthetic_pred_vs_gt_wd_{wds[best_idx]}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating prediction vs ground truth plot: {e}")
        plt.close()
