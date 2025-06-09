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

# Plot accuracy curves
try:
    data = experiment_data["Ablate_Meta_Loss_Ranking"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Ablate_Meta_Loss_Ranking Dataset Accuracy Curves")
    fig.text(
        0.5, 0.92, "Left: Training Accuracy; Right: Validation Accuracy", ha="center"
    )
    for ds, d in data.items():
        epochs = np.arange(len(d["metrics"]["train"]))
        ax1.plot(epochs, d["metrics"]["train"], label=ds)
        ax2.plot(epochs, d["metrics"]["val"], label=ds)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    plt.savefig(
        os.path.join(working_dir, "ablate_meta_loss_ranking_accuracy_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot loss curves
try:
    data = experiment_data["Ablate_Meta_Loss_Ranking"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Ablate_Meta_Loss_Ranking Dataset Loss Curves")
    fig.text(0.5, 0.92, "Left: Training Loss; Right: Validation Loss", ha="center")
    for ds, d in data.items():
        epochs = np.arange(len(d["losses"]["train"]))
        ax1.plot(epochs, d["losses"]["train"], label=ds)
        ax2.plot(epochs, d["losses"]["val"], label=ds)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.savefig(os.path.join(working_dir, "ablate_meta_loss_ranking_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot Spearman correlation history
try:
    data = experiment_data["Ablate_Meta_Loss_Ranking"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Ablate_Meta_Loss_Ranking Spearman Correlation History")
    fig.text(0.5, 0.92, "Spearman correlation per meta-update step", ha="center")
    for ds, d in data.items():
        ax.plot(np.arange(len(d["corrs"])), d["corrs"], label=ds)
    ax.set_xlabel("Meta-update step")
    ax.set_ylabel("Spearman Correlation")
    ax.legend()
    plt.savefig(
        os.path.join(
            working_dir, "ablate_meta_loss_ranking_spearman_correlation_history.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating correlation plot: {e}")
    plt.close()

# Plot N_meta history
try:
    data = experiment_data["Ablate_Meta_Loss_Ranking"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Ablate_Meta_Loss_Ranking N_meta History")
    fig.text(0.5, 0.92, "N_meta per meta-update step", ha="center")
    for ds, d in data.items():
        ax.plot(np.arange(len(d["N_meta_history"])), d["N_meta_history"], label=ds)
    ax.set_xlabel("Meta-update step")
    ax.set_ylabel("N_meta")
    ax.legend()
    plt.savefig(
        os.path.join(working_dir, "ablate_meta_loss_ranking_n_meta_history.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating N_meta history plot: {e}")
    plt.close()
