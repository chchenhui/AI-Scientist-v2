import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Figure 1: Learning Rate Sweep
try:
    data = np.load(
        "experiment_results/experiment_f9213b4ae464430eac366ef28c91a9e1_proc_75765/experiment_data.npy",
        allow_pickle=True
    ).item()
    syn = data["learning_rate"]["synthetic"]
    lrs = syn["params"]
    lt = np.array(syn["losses"]["train"])
    lv = np.array(syn["losses"]["val"])
    at = np.array(syn["metrics"]["train"])
    av = np.array(syn["metrics"]["val"])
    epochs = np.arange(1, lt.shape[1] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for i, lr in enumerate(lrs):
        ax1.plot(epochs, lt[i], label=f"LR={lr} train")
        ax1.plot(epochs, lv[i], "--", label=f"LR={lr} val")
    ax1.set_title("Training and Validation Loss vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    remove_spines(ax1)

    for i, lr in enumerate(lrs):
        ax2.plot(epochs, at[i], label=f"LR={lr} train")
        ax2.plot(epochs, av[i], "--", label=f"LR={lr} val")
    ax2.set_title("Training and Validation AICR vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AICR")
    ax2.legend(loc="lower right")
    remove_spines(ax2)

    fig.suptitle("Learning Rate Sweep on Synthetic Data", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig1_lr_sweep.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"[Fig1] {e}")

# Figure 2: Embedding Dimensionality Ablation
try:
    data = np.load(
        "experiment_results/experiment_36c43541e45e4d679c083af3288ccb05_proc_87823/experiment_data.npy",
        allow_pickle=True
    ).item()
    syn = data["embedding_dim"]["synthetic"]
    dims = syn["params"]
    lt = np.array(syn["losses"]["train"])
    lv = np.array(syn["losses"]["val"])
    at = np.array(syn["metrics"]["train"])
    av = np.array(syn["metrics"]["val"])
    epochs = np.arange(1, lt.shape[1] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for i, d in enumerate(dims):
        ax1.plot(epochs, lt[i], label=f"dim={d} train")
        ax1.plot(epochs, lv[i], "--", label=f"dim={d} val")
    ax1.set_title("Loss vs Epoch for Various Embedding Dimensions")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    remove_spines(ax1)

    for i, d in enumerate(dims):
        ax2.plot(epochs, at[i], label=f"dim={d} train")
        ax2.plot(epochs, av[i], "--", label=f"dim={d} val")
    ax2.set_title("AICR vs Epoch for Various Embedding Dimensions")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AICR")
    ax2.legend(loc="lower right")
    remove_spines(ax2)

    fig.suptitle("Embedding Dimensionality Ablation", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig2_embedding_dim_ablation.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"[Fig2] {e}")

# Figure 3: Weight Decay Ablation
try:
    data = np.load(
        "experiment_results/experiment_aa96c69c87bd4476825267de89318133_proc_87823/experiment_data.npy",
        allow_pickle=True
    ).item()
    syn = data["weight_decay"]["synthetic"]
    wds = syn["params"]
    lt = syn["losses"]["train"]
    lv = syn["losses"]["val"]
    at = syn["metrics"]["train"]
    av = syn["metrics"]["val"]
    epochs = np.arange(1, len(lt[0]) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    for wd, tr, va in zip(wds, lt, lv):
        ax1.plot(epochs, tr, label=f"wd={wd} train")
        ax1.plot(epochs, va, "--", label=f"wd={wd} val")
    ax1.set_title("Loss vs Epoch under Weight Decay")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    remove_spines(ax1)

    for wd, tr, va in zip(wds, at, av):
        ax2.plot(epochs, tr, label=f"wd={wd} train")
        ax2.plot(epochs, va, "--", label=f"wd={wd} val")
    ax2.set_title("AICR vs Epoch under Weight Decay")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AICR")
    ax2.legend(loc="lower right")
    remove_spines(ax2)

    final_av = [v[-1] for v in av]
    ax3.bar([str(wd) for wd in wds], final_av, color="C2")
    ax3.set_title("Final Validation AICR by Weight Decay")
    ax3.set_xlabel("Weight Decay")
    ax3.set_ylabel("AICR")
    remove_spines(ax3)

    fig.suptitle("Weight Decay Ablation Study", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig3_weight_decay_ablation.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"[Fig3] {e}")

# Figure 4: Fixed Random Embedding Ablation
try:
    data = np.load(
        "experiment_results/experiment_8c4e8faf60cf4c30af8e2e9f9013c1a1_proc_87824/experiment_data.npy",
        allow_pickle=True
    ).item()
    syn = data["fixed_random_embedding"]["synthetic"]
    lrs = syn["params"]
    lt = syn["losses"]["train"]
    lv = syn["losses"]["val"]
    st = syn["metrics"]["train"]
    sv = syn["metrics"]["val"]
    epochs = np.arange(1, len(lt[0]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for lr, tr, va in zip(lrs, lt, lv):
        ax1.plot(epochs, tr, label=f"LR={lr} train")
        ax1.plot(epochs, va, "--", label=f"LR={lr} val")
    ax1.set_title("Loss vs Epoch for Fixed Random Embedding")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    remove_spines(ax1)

    for lr, tr, va in zip(lrs, st, sv):
        ax2.plot(epochs, tr, label=f"LR={lr} train")
        ax2.plot(epochs, va, "--", label=f"LR={lr} val")
    ax2.set_title("Success Rate vs Epoch for Fixed Random Embedding")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Success Rate")
    ax2.legend(loc="lower right")
    remove_spines(ax2)

    fig.suptitle("Fixed Random Embedding Ablation", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig4_fixed_random_embedding.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"[Fig4] {e}")