import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for ds_name, ds_data in experiment_data.get("no_layernorm", {}).items():
    losses = ds_data.get("losses", {})
    metrics = ds_data.get("metrics", {})

    # Loss curve
    try:
        plt.figure()
        tr, va = losses.get("train"), losses.get("val")
        if tr and va:
            epochs = range(1, len(tr) + 1)
            plt.plot(epochs, tr, marker="o", label="Train Loss")
            plt.plot(epochs, va, marker="o", label="Val Loss")
            plt.title(f"{ds_name} Loss Curve\nTraining vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # Memory Retention Ratio
    try:
        plt.figure()
        mr = metrics.get("Memory Retention Ratio", {})
        tr_mr, va_mr = mr.get("train"), mr.get("val")
        if tr_mr and va_mr:
            epochs = range(1, len(tr_mr) + 1)
            plt.plot(epochs, tr_mr, marker="o", label="Train MRR")
            plt.plot(epochs, va_mr, marker="o", label="Val MRR")
            plt.title(f"{ds_name} Memory Retention Ratio\nTraining vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Ratio")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_memory_retention_ratio.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating MRR plot for {ds_name}: {e}")
        plt.close()

    # Entropy-Weighted Memory Efficiency
    try:
        plt.figure()
        eme = metrics.get("Entropy-Weighted Memory Efficiency", {})
        tr_eme, va_eme = eme.get("train"), eme.get("val")
        if tr_eme and va_eme:
            epochs = range(1, len(tr_eme) + 1)
            plt.plot(epochs, tr_eme, marker="o", label="Train EME")
            plt.plot(epochs, va_eme, marker="o", label="Val EME")
            plt.title(
                f"{ds_name} Entropy-Weighted Memory Efficiency\nTraining vs Validation"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Efficiency")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_entropy_memory_efficiency.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating EME plot for {ds_name}: {e}")
        plt.close()

    # Predictions vs Ground Truth
    preds = ds_data.get("predictions", [])
    gts = ds_data.get("ground_truth", [])
    for i in range(min(5, len(preds))):
        try:
            gt_str = "".join(chr(c) for c in gts[i])
            pred_str = "".join(chr(c) for c in preds[i])
            plt.figure(figsize=(8, 4))
            plt.axis("off")
            plt.text(0.05, 0.6, gt_str, wrap=True)
            plt.text(0.05, 0.4, pred_str, wrap=True)
            plt.title(
                f"{ds_name} Predictions vs Ground Truth (Sample {i})\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.savefig(os.path.join(working_dir, f"{ds_name}_pred_vs_gt_{i}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating prediction plot for {ds_name} sample {i}: {e}")
            plt.close()

    # Print final metrics
    final_loss = losses.get("val")[-1] if losses.get("val") else None
    final_mrr = metrics.get("Memory Retention Ratio", {}).get("val", [None])[-1]
    final_eme = metrics.get("Entropy-Weighted Memory Efficiency", {}).get(
        "val", [None]
    )[-1]
    print(
        f"{ds_name}: Final Val Loss = {final_loss}, MRR = {final_mrr}, EME = {final_eme}"
    )
