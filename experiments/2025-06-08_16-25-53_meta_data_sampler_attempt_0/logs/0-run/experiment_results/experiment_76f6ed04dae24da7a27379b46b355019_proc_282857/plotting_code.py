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
else:
    for ab_name, datasets in experiment_data.items():
        for dataset_name, d in datasets.items():
            # Print final metrics
            try:
                tr_acc = d["metrics"]["train"][-1]
                val_acc = d["metrics"]["val"][-1]
                tr_loss = d["losses"]["train"][-1]
                val_loss = d["losses"]["val"][-1]
                print(
                    f"{dataset_name} final train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            except Exception as e:
                print(f"Error printing metrics for {dataset_name}: {e}")

            epochs = list(range(1, len(d["losses"]["train"]) + 1))

            # Loss curves
            try:
                plt.figure()
                plt.plot(epochs, d["losses"]["train"], label="Train Loss")
                plt.plot(epochs, d["losses"]["val"], label="Val Loss")
                plt.title(f"Loss Curves for {dataset_name}\nTrain vs Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curve.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating loss curve for {dataset_name}: {e}")
                plt.close()

            # Accuracy curves
            try:
                plt.figure()
                plt.plot(epochs, d["metrics"]["train"], label="Train Acc")
                plt.plot(epochs, d["metrics"]["val"], label="Val Acc")
                plt.title(
                    f"Accuracy Curves for {dataset_name}\nTrain vs Validation Accuracy"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig(
                    os.path.join(working_dir, f"{dataset_name}_accuracy_curve.png")
                )
                plt.close()
            except Exception as e:
                print(f"Error creating accuracy curve for {dataset_name}: {e}")
                plt.close()

            # Spearman correlation history
            try:
                if d.get("corrs"):
                    plt.figure()
                    plt.plot(range(1, len(d["corrs"]) + 1), d["corrs"], marker="o")
                    plt.title(
                        f"Spearman Correlation History for {dataset_name}\nMeta-Learning Correlations"
                    )
                    plt.xlabel("Meta Update Step")
                    plt.ylabel("Spearman r")
                    plt.savefig(
                        os.path.join(working_dir, f"{dataset_name}_corr_history.png")
                    )
                    plt.close()
            except Exception as e:
                print(f"Error creating correlation history for {dataset_name}: {e}")
                plt.close()

            # N_meta history
            try:
                if d.get("N_meta_history"):
                    plt.figure()
                    plt.plot(
                        range(1, len(d["N_meta_history"]) + 1),
                        d["N_meta_history"],
                        marker="o",
                    )
                    plt.title(
                        f"N_meta History for {dataset_name}\nMeta Batch Size Evolution"
                    )
                    plt.xlabel("Meta Update Step")
                    plt.ylabel("N_meta")
                    plt.savefig(
                        os.path.join(working_dir, f"{dataset_name}_N_meta_history.png")
                    )
                    plt.close()
            except Exception as e:
                print(f"Error creating N_meta history for {dataset_name}: {e}")
                plt.close()

            # Label distribution: ground truth vs predictions
            try:
                preds = d["predictions"][0]
                gt = d["ground_truth"][0]
                labels = np.unique(np.concatenate([gt, preds]))
                counts_gt = [np.sum(gt == l) for l in labels]
                counts_pred = [np.sum(preds == l) for l in labels]
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].bar(labels, counts_gt)
                axes[0].set_title("Ground Truth")
                axes[1].bar(labels, counts_pred)
                axes[1].set_title("Predictions")
                fig.suptitle(f"Label Distribution for {dataset_name}")
                fig.text(
                    0.5, 0.94, "Left: Ground Truth, Right: Predictions", ha="center"
                )
                plt.savefig(
                    os.path.join(working_dir, f"{dataset_name}_label_distribution.png")
                )
                plt.close()
            except Exception as e:
                print(f"Error creating label distribution for {dataset_name}: {e}")
                plt.close()
