import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
else:
    # Print final evaluation metrics
    print("Final evaluation metrics:")
    for pool, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            try:
                acc = data["metrics"]["val"][-1]
                mai = data["mai"][-1]
                print(f"{pool} | {ds_name} | val_acc: {acc:.4f}, MAI: {mai:.4f}")
            except:
                pass

    # Generate plots
    for pool, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            epochs = np.arange(1, len(data["losses"]["train"]) + 1)
            # Loss & Accuracy curves
            try:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.plot(epochs, data["losses"]["train"], label="Train Loss")
                plt.plot(epochs, data["losses"]["val"], label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{pool} | {ds_name} Loss")
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(epochs, data["metrics"]["train"], label="Train Acc")
                plt.plot(epochs, data["metrics"]["val"], label="Val Acc")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"{pool} | {ds_name} Accuracy")
                plt.legend()
                plt.suptitle("Loss (Left) vs Accuracy (Right)")
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, f"{pool}_{ds_name}_loss_acc.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating loss/acc plot for {pool}-{ds_name}: {e}")
                plt.close()

            # Alignment & MAI curves
            try:
                plt.figure()
                plt.plot(epochs, data["alignments"]["train"], label="Train Align")
                plt.plot(epochs, data["alignments"]["val"], label="Val Align")
                plt.plot(epochs, data["mai"], label="MAI", linestyle="--")
                plt.xlabel("Epoch")
                plt.ylabel("Value")
                plt.title(f"{pool} | {ds_name} Alignment & MAI")
                plt.legend()
                plt.suptitle("Alignment & MAI over Epochs")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{pool}_{ds_name}_align_mai.png")
                )
                plt.close()
            except Exception as e:
                print(f"Error creating alignment/MAI plot for {pool}-{ds_name}: {e}")
                plt.close()

            # Predictions vs Ground Truth bar chart
            try:
                gt = data["ground_truth"]
                preds = data["predictions"]
                n_classes = max(gt.max(), preds.max()) + 1
                gt_counts = np.bincount(gt, minlength=n_classes)
                pred_counts = np.bincount(preds, minlength=n_classes)
                x = np.arange(n_classes)
                width = 0.35
                fig, ax = plt.subplots()
                ax.bar(x - width / 2, gt_counts, width, label="Ground Truth")
                ax.bar(x + width / 2, pred_counts, width, label="Predictions")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                ax.set_xticks(x)
                ax.set_title(f"{pool} | {ds_name} Preds vs GT")
                fig.suptitle("Left: Ground Truth, Right: Predictions")
                ax.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{pool}_{ds_name}_preds_vs_gt.png")
                )
                plt.close()
            except Exception as e:
                print(f"Error creating preds vs gt plot for {pool}-{ds_name}: {e}")
                plt.close()
