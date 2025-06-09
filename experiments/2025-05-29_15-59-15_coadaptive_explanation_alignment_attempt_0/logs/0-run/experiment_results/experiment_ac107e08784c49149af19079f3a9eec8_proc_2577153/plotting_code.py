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

tpn = experiment_data.get("teacher_prob_noise", {})
preferred_key = "ai_bs_32_user_bs_32"

for skey, content in tpn.items():
    sigma = skey.split("_", 1)[1]
    bs_dict = content.get("batch_size", {})
    key_plot = (
        preferred_key
        if preferred_key in bs_dict
        else (next(iter(bs_dict)) if bs_dict else None)
    )
    if key_plot is None:
        continue
    data = bs_dict[key_plot]
    metrics = data.get("metrics", {})
    train_acc = metrics.get("train", [])
    val_acc = metrics.get("val", [])
    alignment_scores = data.get("alignment_scores", [])

    try:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(train_acc, label="Train Acc")
        axs[0].plot(val_acc, label="Val Acc")
        axs[0].set_title(f"Accuracy Curves\nσ={sigma}")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()
        axs[1].plot(alignment_scores, color="tab:green")
        axs[1].set_title(f"Alignment Scores\nσ={sigma}")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Alignment")
        fig.suptitle(f"teacher_prob_noise σ={sigma} | {key_plot}")
        out_path = os.path.join(
            working_dir, f"teacher_prob_noise_sigma_{sigma}_{key_plot}_curves.png"
        )
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for σ={sigma}: {e}")
        plt.close()
