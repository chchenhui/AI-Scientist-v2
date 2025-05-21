import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = ["ag_news", "yelp_polarity", "dbpedia_14"]
ablation_types = ["random", "importance"]
head_counts_list = [2, 4, 8, 12]

# 1–3: MAI vs head count for each dataset
for name in datasets:
    try:
        plt.figure()
        for t in ablation_types:
            heads = []
            mais = []
            data = experiment_data.get(t, {}).get(name, {})
            hs = data.get("head_counts", [])
            m = data.get("mai", [])
            # compute final MAI per head count
            for h in sorted(set(hs)):
                idxs = [i for i, hh in enumerate(hs) if hh == h]
                if idxs:
                    heads.append(h)
                    mais.append(m[max(idxs)])
            # sort pairs
            paired = sorted(zip(heads, mais))
            x, y = zip(*paired)
            plt.plot(x, y, marker="o", label=t.capitalize())
        plt.xlabel("Number of Heads")
        plt.ylabel("MAI")
        plt.title(f"{name} MAI vs Head Count\nRandom vs Importance Ablation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_MAI_vs_heads.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MAI plot for {name}: {e}")
        plt.close()

# 4–5: Loss curves for ag_news at head_count=12
for t in ablation_types:
    try:
        plt.figure()
        data = experiment_data.get(t, {}).get("ag_news", {})
        hs = data.get("head_counts", [])
        train_losses = data.get("losses", {}).get("train", [])
        val_losses = data.get("losses", {}).get("val", [])
        # filter for head_count = 12
        idxs = [i for i, hh in enumerate(hs) if hh == 12]
        epochs = list(range(1, len(idxs) + 1))
        tr = [train_losses[i] for i in idxs]
        vl = [val_losses[i] for i in idxs]
        plt.plot(epochs, tr, marker="o", label="Train Loss")
        plt.plot(epochs, vl, marker="o", linestyle="--", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"ag_news Loss Curves for {t.capitalize()} Ablation\nHead Count = 12")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"ag_news_{t}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for ag_news ({t}): {e}")
        plt.close()
