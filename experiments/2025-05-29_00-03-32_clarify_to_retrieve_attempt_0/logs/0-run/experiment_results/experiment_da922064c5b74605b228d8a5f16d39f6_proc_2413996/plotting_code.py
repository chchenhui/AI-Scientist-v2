import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}
mpaf = data.get("multi_passage_answer_fusion", {})

# Print validation metrics
for ds, ds_data in mpaf.items():
    val = ds_data.get("metrics", {}).get("val", [])
    if len(val) >= 2:
        wf, nf = val[0], val[1]
        print(f"{ds} WITH FUSION: {wf}")
        print(f"{ds} NO FUSION:    {nf}")

# Plotting
for ds, ds_data in mpaf.items():
    val = ds_data.get("metrics", {}).get("val", [])
    if len(val) < 2:
        continue
    wf, nf = val[0], val[1]
    # Accuracy comparison
    try:
        plt.figure()
        labels = ["Baseline Acc", "Clar Acc"]
        x = np.arange(len(labels))
        w = 0.35
        plt.bar(x - w / 2, [wf["baseline_acc"], wf["clar_acc"]], w, label="With Fusion")
        plt.bar(x + w / 2, [nf["baseline_acc"], nf["clar_acc"]], w, label="No Fusion")
        plt.xticks(x, labels)
        plt.ylabel("Accuracy")
        plt.title(f"{ds} - Accuracy Comparison\nWith Fusion vs No Fusion")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_accuracy_comparison.png"))
    except Exception as e:
        print(f"Error creating accuracy plot for {ds}: {e}")
    finally:
        plt.close()

    # CES comparison
    try:
        plt.figure()
        x = np.arange(1)
        w = 0.35
        plt.bar(x - w / 2, [wf["CES"]], w, label="With Fusion")
        plt.bar(x + w / 2, [nf["CES"]], w, label="No Fusion")
        plt.xticks(x, ["CES"])
        plt.ylabel("CES")
        plt.title(f"{ds} - CES Comparison\nWith Fusion vs No Fusion")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_CES_comparison.png"))
    except Exception as e:
        print(f"Error creating CES plot for {ds}: {e}")
    finally:
        plt.close()

# Training/validation curves if available
for ds, ds_data in mpaf.items():
    metrics = ds_data.get("metrics", {})
    train = metrics.get("train", [])
    val = metrics.get("val", [])
    if train and val:
        try:
            plt.figure()
            epochs = np.arange(1, len(train) + 1)
            tr = [m["baseline_acc"] for m in train]
            vl = [m["baseline_acc"] for m in val[: len(epochs)]]
            plt.plot(epochs, tr, label="Train Baseline Acc")
            plt.plot(epochs, vl, label="Val Baseline Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Baseline Accuracy")
            plt.title(f"{ds} - Baseline Acc over Epochs")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds}_baseline_acc_curve.png"))
        except Exception as e:
            print(f"Error creating training curve for {ds}: {e}")
        finally:
            plt.close()
