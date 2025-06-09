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
    # Per-dataset visualizations
    for dataset, exp in experiment_data.items():
        # Loss curves
        try:
            plt.figure()
            epochs = [d["epoch"] for d in exp["losses"]["train"]]
            tr_loss = [d["loss"] for d in exp["losses"]["train"]]
            vl_loss = [d["loss"] for d in exp["losses"]["val"]]
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, vl_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} Loss Curve\nTrain vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curve.png"))
        except Exception as e:
            print(f"Error creating loss plot for {dataset}: {e}")
        finally:
            plt.close()

        # Detection AUC curves
        try:
            plt.figure()
            det = exp["metrics"]["detection"]
            epochs = [d["epoch"] for d in det]
            auc_v = [d["auc_vote"] for d in det]
            auc_k = [d["auc_kl"] for d in det]
            plt.plot(epochs, auc_v, label="Vote AUC")
            plt.plot(epochs, auc_k, label="KL AUC")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.title(f"{dataset} Detection AUC Curve\nLeft: Vote, Right: KL")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_detection_auc_curve.png"))
        except Exception as e:
            print(f"Error creating detection AUC plot for {dataset}: {e}")
        finally:
            plt.close()

        # Class distribution bar chart
        try:
            plt.figure()
            gt = np.array(exp["ground_truth"])
            preds = np.array(exp["predictions"])
            classes = sorted(set(np.concatenate((gt, preds))))
            counts_gt = [np.sum(gt == c) for c in classes]
            counts_pred = [np.sum(preds == c) for c in classes]
            x = np.arange(len(classes))
            width = 0.35
            plt.bar(x - width / 2, counts_gt, width, label="Ground Truth")
            plt.bar(x + width / 2, counts_pred, width, label="Predicted")
            plt.xticks(x, [str(c) for c in classes])
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                f"{dataset} Class Distribution\nLeft: Ground Truth, Right: Predicted"
            )
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_val_class_distribution.png")
            )
        except Exception as e:
            print(f"Error creating class distribution plot for {dataset}: {e}")
        finally:
            plt.close()

    # Comparison across datasets
    try:
        plt.figure()
        datasets = list(experiment_data.keys())
        final_vote = [
            experiment_data[d]["metrics"]["detection"][-1]["auc_vote"] for d in datasets
        ]
        final_kl = [
            experiment_data[d]["metrics"]["detection"][-1]["auc_kl"] for d in datasets
        ]
        x = np.arange(len(datasets))
        width = 0.35
        plt.bar(x - width / 2, final_vote, width, label="Vote AUC")
        plt.bar(x + width / 2, final_kl, width, label="KL AUC")
        plt.xticks(x, datasets)
        plt.xlabel("Dataset")
        plt.ylabel("AUC")
        plt.title(
            "Final Detection AUC Comparison Across Datasets\nLeft: Vote, Right: KL Across Datasets"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_final_detection_auc.png"))
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
    finally:
        plt.close()

    # Print final detection metrics
    for dataset, exp in experiment_data.items():
        final = exp["metrics"]["detection"][-1]
        print(
            f"{dataset}: Final Detection AUC_vote={final['auc_vote']:.4f}, AUC_kl={final['auc_kl']:.4f}"
        )
