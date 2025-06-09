import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for head_agg, ds_dict in experiment_data.items():
    for ds_name, ds_data in ds_dict.items():
        epochs = range(1, len(ds_data["losses"]["train"]) + 1)
        # Loss curves
        try:
            plt.figure()
            plt.plot(epochs, ds_data["losses"]["train"], label="Train Loss")
            plt.plot(epochs, ds_data["losses"]["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} ({head_agg}) Loss Curves")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_{head_agg}_loss.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {head_agg}-{ds_name}: {e}")
            plt.close()
        # Memory Retention Ratio
        try:
            plt.figure()
            plt.plot(
                epochs,
                ds_data["metrics"]["Memory Retention Ratio"]["train"],
                label="Train MRR",
            )
            plt.plot(
                epochs,
                ds_data["metrics"]["Memory Retention Ratio"]["val"],
                label="Val MRR",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Memory Retention Ratio")
            plt.title(f"{ds_name} ({head_agg}) Memory Retention Ratio")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_{head_agg}_mrr.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating MRR plot for {head_agg}-{ds_name}: {e}")
            plt.close()
        # Entropy-Weighted Memory Efficiency
        try:
            plt.figure()
            plt.plot(
                epochs,
                ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]["train"],
                label="Train EME",
            )
            plt.plot(
                epochs,
                ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"],
                label="Val EME",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Entropy-Weighted Memory Efficiency")
            plt.title(f"{ds_name} ({head_agg}) EME")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_{head_agg}_eme.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating EME plot for {head_agg}-{ds_name}: {e}")
            plt.close()
        # Sample comparison
        try:
            gt_seq = ds_data.get("ground_truth", [[]])[0][:50]
            pred_seq = ds_data.get("predictions", [[]])[0][:50]
            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.plot(gt_seq, marker="o")
            plt.title("Ground Truth")
            plt.xlabel("Token Index")
            plt.ylabel("Token ID")
            plt.subplot(1, 2, 2)
            plt.plot(pred_seq, marker="x")
            plt.title("Generated Samples")
            plt.suptitle(
                f"{ds_name} ({head_agg}) - Left: Ground Truth, Right: Generated Samples"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_{head_agg}_sample_comp.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating sample comparison for {head_agg}-{ds_name}: {e}")
            plt.close()
