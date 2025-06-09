import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# extract synthetic results
ed = experiment_data.get("dict_orth", {}).get("synthetic", {})
train_metrics = ed.get("metrics", {}).get("train", [])
val_metrics = ed.get("metrics", {}).get("val", [])
loss_train = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])
predictions = ed.get("predictions", [])
ground_truth = ed.get("ground_truth", [])

# define lambda2 grid
lam2_list = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
lam2_str = ["0", "1e-4", "1e-3", "1e-2", "1e-1"]

# compute and print final test errors
if val_metrics:
    final_errs = {lam: round(vals[-1], 4) for lam, vals in zip(lam2_str, val_metrics)}
    print("Final test relative errors per λ2:")
    for lam, err in final_errs.items():
        print(f"  λ2={lam}: {err}")

# plot relative error curves
try:
    plt.figure()
    for lam, lam_s, tr, vl in zip(lam2_list, lam2_str, train_metrics, val_metrics):
        plt.plot(tr, label=f"Train λ2={lam_s}")
        plt.plot(vl, "--", label=f"Val   λ2={lam_s}")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.title("Synthetic Dataset: Training & Validation Relative Error Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves: {e}")
    plt.close()

# plot loss curves
try:
    plt.figure()
    for lam, lam_s, lt, lv in zip(lam2_list, lam2_str, loss_train, loss_val):
        plt.plot(lt, label=f"Train λ2={lam_s}")
        plt.plot(lv, "--", label=f"Val   λ2={lam_s}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Synthetic Dataset: Training & Validation Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# reconstruction examples for three λ2 values
for idx in [0, 2, 4]:
    lam_s = lam2_str[idx]
    try:
        gt = ground_truth[idx][0].reshape(32, 32)
        pred = predictions[idx][0].reshape(32, 32)
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(gt, cmap="viridis")
        plt.title("Ground Truth")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap="viridis")
        plt.title("Reconstructed")
        plt.axis("off")
        plt.suptitle(
            f"Synthetic Dataset; Left: Ground Truth, Right: Reconstructed Samples (λ2={lam_s})"
        )
        fname = f"synthetic_recon_lambda2_{lam_s}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating recon plot for λ2={lam_s}: {e}")
        plt.close()
