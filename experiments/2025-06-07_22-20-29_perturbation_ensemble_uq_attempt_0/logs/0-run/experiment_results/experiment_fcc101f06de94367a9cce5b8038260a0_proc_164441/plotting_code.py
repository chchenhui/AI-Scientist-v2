import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Group by dataset and variant
dataset_groups = {}
for key in experiment_data:
    if key.endswith("_baseline"):
        ds = key[:-9]
        dataset_groups.setdefault(ds, {})["baseline"] = key
    elif key.endswith("_paraphrase_aug"):
        ds = key[:-15]
        dataset_groups.setdefault(ds, {})["paraphrase_aug"] = key

# Print final validation metrics
for ds, keys in dataset_groups.items():
    for var, key in keys.items():
        mets = experiment_data[key]["metrics"]["val"]
        if mets:
            last = mets[-1]
            print(
                f"{ds} {var} final AUC_vote: {last['auc_vote']:.3f}, AUC_kl: {last['auc_kl']:.3f}"
            )

# Plot training vs validation loss
try:
    plt.figure()
    for ds, keys in dataset_groups.items():
        if "baseline" in keys:
            k = keys["baseline"]
            tr = [e["loss"] for e in experiment_data[k]["losses"]["train"]]
            va = [e["loss"] for e in experiment_data[k]["losses"]["val"]]
            ep = [e["epoch"] for e in experiment_data[k]["losses"]["train"]]
            plt.plot(ep, tr, label=f"{ds} baseline train")
            plt.plot(ep, va, label=f"{ds} baseline val")
        if "paraphrase_aug" in keys:
            k = keys["paraphrase_aug"]
            tr = [e["loss"] for e in experiment_data[k]["losses"]["train"]]
            va = [e["loss"] for e in experiment_data[k]["losses"]["val"]]
            ep = [e["epoch"] for e in experiment_data[k]["losses"]["train"]]
            plt.plot(ep, tr, linestyle="--", label=f"{ds} paraphrase train")
            plt.plot(ep, va, linestyle="--", label=f"{ds} paraphrase val")
    plt.title("Training vs Validation Loss for All Datasets")
    plt.suptitle("Solid: baseline, Dashed: paraphrase_aug; Train vs. Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot AUC_vote over epochs
try:
    plt.figure()
    for ds, keys in dataset_groups.items():
        if "baseline" in keys:
            k = keys["baseline"]
            met = experiment_data[k]["metrics"]["val"]
            ep = [m["epoch"] for m in met]
            auc = [m["auc_vote"] for m in met]
            plt.plot(ep, auc, label=f"{ds} baseline")
        if "paraphrase_aug" in keys:
            k = keys["paraphrase_aug"]
            met = experiment_data[k]["metrics"]["val"]
            ep = [m["epoch"] for m in met]
            auc = [m["auc_vote"] for m in met]
            plt.plot(ep, auc, linestyle="--", label=f"{ds} paraphrase")
    plt.title("AUC_vote over Epochs for All Datasets")
    plt.suptitle("Solid: baseline, Dashed: paraphrase_aug")
    plt.xlabel("Epoch")
    plt.ylabel("AUC_vote")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_auc_vote_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AUC_vote plot: {e}")
    plt.close()

# Plot AUC_kl over epochs
try:
    plt.figure()
    for ds, keys in dataset_groups.items():
        if "baseline" in keys:
            k = keys["baseline"]
            met = experiment_data[k]["metrics"]["val"]
            ep = [m["epoch"] for m in met]
            auc = [m["auc_kl"] for m in met]
            plt.plot(ep, auc, label=f"{ds} baseline")
        if "paraphrase_aug" in keys:
            k = keys["paraphrase_aug"]
            met = experiment_data[k]["metrics"]["val"]
            ep = [m["epoch"] for m in met]
            auc = [m["auc_kl"] for m in met]
            plt.plot(ep, auc, linestyle="--", label=f"{ds} paraphrase")
    plt.title("AUC_kl over Epochs for All Datasets")
    plt.suptitle("Solid: baseline, Dashed: paraphrase_aug")
    plt.xlabel("Epoch")
    plt.ylabel("AUC_kl")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_auc_kl_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AUC_kl plot: {e}")
    plt.close()
