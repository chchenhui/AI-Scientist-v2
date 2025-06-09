import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ablate = data.get("Ablate_Meta_Inner_Update_Steps", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ablate = {}

for dataset, exp in ablate.items():
    # accuracy plot
    try:
        plt.figure()
        acc = exp["metrics"]
        plt.plot(acc["train"], label="Train")
        plt.plot(acc["val"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset}: Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_accuracy.png"))
    except Exception as e:
        print(f"Error creating accuracy plot for {dataset}: {e}")
    finally:
        plt.close()
    # loss plot
    try:
        plt.figure()
        loss = exp["losses"]
        plt.plot(loss["train"], label="Train")
        plt.plot(loss["val"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_loss.png"))
    except Exception as e:
        print(f"Error creating loss plot for {dataset}: {e}")
    finally:
        plt.close()
    # spearman correlation plot
    try:
        plt.figure()
        corrs = exp.get("corrs", [])
        plt.plot(corrs, marker="o")
        plt.xlabel("Meta‐Update Step")
        plt.ylabel("Spearman Correlation")
        plt.title(f"{dataset}: Spearman Corr over Meta‐Updates")
        plt.savefig(os.path.join(working_dir, f"{dataset}_spearman_corr.png"))
    except Exception as e:
        print(f"Error creating spearman plot for {dataset}: {e}")
    finally:
        plt.close()
    # N_meta history plot
    try:
        plt.figure()
        nmh = exp.get("N_meta_history", [])
        plt.plot(nmh, marker="s")
        plt.xlabel("Meta‐Update Step")
        plt.ylabel("N_meta")
        plt.title(f"{dataset}: N_meta History")
        plt.savefig(os.path.join(working_dir, f"{dataset}_n_meta_history.png"))
    except Exception as e:
        print(f"Error creating N_meta plot for {dataset}: {e}")
    finally:
        plt.close()
