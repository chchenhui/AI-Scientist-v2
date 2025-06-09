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
    best_aucs = {}
    for dataset, exp in experiment_data.items():
        # find best hyperparams by val AUC
        best_entry = max(exp["metrics"]["val"], key=lambda x: x["auc"])
        bs, lr = best_entry["bs"], best_entry["lr"]
        best_aucs[dataset] = best_entry["auc"]
        print(f"{dataset}: Best Val AUC = {best_entry['auc']:.4f} at bs={bs}, lr={lr}")
        # collect losses
        loss_train = [
            (d["epoch"], d["loss"])
            for d in exp["losses"]["train"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        loss_val = [
            (d["epoch"], d["loss"])
            for d in exp["losses"]["val"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        loss_train.sort()
        loss_val.sort()
        epochs = [e for e, _ in loss_train]
        tr_loss = [l for _, l in loss_train]
        vl_loss = [l for _, l in loss_val]
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, vl_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} Loss Curve (Train vs Validation)\nbs={bs}, lr={lr}")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_loss_curve_bs{bs}_lr{lr}.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dataset}: {e}")
            plt.close()
        # collect AUCs
        auc_train = [
            (d["epoch"], d["auc"])
            for d in exp["metrics"]["train"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        auc_val = [
            (d["epoch"], d["auc"])
            for d in exp["metrics"]["val"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        auc_train.sort()
        auc_val.sort()
        epochs = [e for e, _ in auc_train]
        tr_auc = [a for _, a in auc_train]
        vl_auc = [a for _, a in auc_val]
        try:
            plt.figure()
            plt.plot(epochs, tr_auc, label="Train AUC")
            plt.plot(epochs, vl_auc, label="Val AUC")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.title(f"{dataset} AUC Curve (Train vs Validation)\nbs={bs}, lr={lr}")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_auc_curve_bs{bs}_lr{lr}.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating AUC plot for {dataset}: {e}")
            plt.close()
        # dataset-specific: synthetic prediction histogram
        if dataset == "synthetic":
            last_epoch = max(
                d["epoch"]
                for d in exp["predictions"]
                if d["bs"] == bs and d["lr"] == lr
            )
            preds = next(
                d["preds"]
                for d in exp["predictions"]
                if d["bs"] == bs and d["lr"] == lr and d["epoch"] == last_epoch
            )
            labels = next(
                d["labels"]
                for d in exp["ground_truth"]
                if d["bs"] == bs and d["lr"] == lr and d["epoch"] == last_epoch
            )
            try:
                plt.figure()
                plt.hist(
                    [preds[labels == 0], preds[labels == 1]],
                    bins=20,
                    label=["Class 0", "Class 1"],
                    alpha=0.7,
                )
                plt.xlabel("Predicted Probability")
                plt.ylabel("Count")
                plt.title("Synthetic Dataset Predictions\nHistogram by True Class")
                plt.legend()
                plt.savefig(
                    os.path.join(working_dir, f"{dataset}_pred_hist_bs{bs}_lr{lr}.png")
                )
                plt.close()
            except Exception as e:
                print(f"Error creating prediction histogram for {dataset}: {e}")
                plt.close()
    # comparison bar chart of best val AUCs
    try:
        plt.figure()
        ds = list(best_aucs.keys())
        vals = [best_aucs[d] for d in ds]
        plt.bar(ds, vals)
        plt.ylabel("Best Val AUC")
        plt.title("Comparison of Best Validation AUCs\nAcross Datasets")
        plt.savefig(os.path.join(working_dir, "comparison_best_val_auc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison AUC bar chart: {e}")
        plt.close()
    # overlay validation AUC curves across datasets
    try:
        plt.figure()
        for dataset, exp in experiment_data.items():
            best = max(exp["metrics"]["val"], key=lambda x: x["auc"])
            bs, lr = best["bs"], best["lr"]
            vals = sorted(
                [
                    (d["epoch"], d["auc"])
                    for d in exp["metrics"]["val"]
                    if d["bs"] == bs and d["lr"] == lr
                ]
            )
            epochs = [e for e, _ in vals]
            aucs = [a for _, a in vals]
            plt.plot(epochs, aucs, label=dataset)
        plt.xlabel("Epoch")
        plt.ylabel("Val AUC")
        plt.title("Validation AUC Curves\nComparison Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "datasets_val_auc_curves_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison AUC curves plot: {e}")
        plt.close()
