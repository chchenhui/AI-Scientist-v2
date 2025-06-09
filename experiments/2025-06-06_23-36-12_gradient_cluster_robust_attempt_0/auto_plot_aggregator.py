import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("figures", exist_ok=True)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 1) Learning‐Rate Sweep: Worst‐Group Accuracy & Loss (2×2)
try:
    path = ("experiment_results/"
            "experiment_be4523f06c524c6bbaf8939619561351_proc_3804/"
            "experiment_data.npy")
    lr_data = np.load(path, allow_pickle=True).item()["learning_rate"]["synthetic"]
    lrs = lr_data["lrs"]
    ta = lr_data["metrics"]["train"]
    va = lr_data["metrics"]["val"]
    tl = lr_data["losses"]["train"]
    vl = lr_data["losses"]["val"]
    epochs = np.arange(1, ta.shape[1] + 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # Worst‐Group Accuracy
    for i, lr in enumerate(lrs):
        axs[0,0].plot(epochs, ta[i], label=f"lr={lr:.0e}")
        axs[0,1].plot(epochs, va[i], label=f"lr={lr:.0e}")
    axs[0,0].set_title("Training Worst-Group Accuracy")
    axs[0,1].set_title("Validation Worst-Group Accuracy")
    # Loss
    for i, lr in enumerate(lrs):
        axs[1,0].plot(epochs, tl[i], label=f"lr={lr:.0e}")
        axs[1,1].plot(epochs, vl[i], label=f"lr={lr:.0e}")
    axs[1,0].set_title("Training Loss")
    axs[1,1].set_title("Validation Loss")
    # Formatting
    for ax in axs.flatten():
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=10)
        despine(ax)
    fig.suptitle("Synthetic Dataset: Learning-Rate Sweep", y=0.95)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig("figures/fig1_lr_sweep.png")
    plt.close(fig)
except Exception as e:
    print("Error in fig1:", e)

# 2) Cluster-Count Variation at Best LR (1×2)
try:
    path = ("experiment_results/"
            "experiment_747fec2aaf314e33a8225f037d912c14_proc_17029/"
            "experiment_data.npy")
    cc = np.load(path, allow_pickle=True).item()["CLUSTER_COUNT_VARIATION"]["synthetic"]
    ks = cc["cluster_counts"]
    blk = cc["learning_rate"]
    lrs_cc = blk["lrs"]
    train_m = blk["metrics"]["train"]
    val_m   = blk["metrics"]["val"]
    epochs = np.arange(1, train_m.shape[2] + 1)
    # pick highest learning rate
    best_idx = np.argmax(lrs_cc)
    selected_k = [2, 4, 8]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for k in selected_k:
        idx = int(np.where(ks == k)[0][0])
        axs[0].plot(epochs, train_m[idx, best_idx], label=f"k={k}")
        axs[1].plot(epochs, val_m[idx, best_idx],   label=f"k={k}")
    axs[0].set_title(f"Train Worst-Group Accuracy\n(lr={lrs_cc[best_idx]:.0e})")
    axs[1].set_title(f"Validation Worst-Group Accuracy\n(lr={lrs_cc[best_idx]:.0e})")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=10)
        despine(ax)
    fig.suptitle("Synthetic: Cluster-Count Variation at Best Learning Rate", y=0.95)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig("figures/fig2_cluster_count_best_lr.png")
    plt.close(fig)
except Exception as e:
    print("Error in fig2:", e)

# 3) Reweighting Strategies: Input-Feature vs Group-Inverse (1×2)
try:
    # Input-feature
    p1 = ("experiment_results/"
          "experiment_7d2ceea87c7346a2aa42a75794b7b182_proc_17029/"
          "experiment_data.npy")
    inf = np.load(p1, allow_pickle=True).item()["INPUT_FEATURE_CLUSTER_REWEIGHTING"]["synthetic"]
    lrs1 = inf["lrs"]
    ta1, va1 = inf["metrics"]["train"], inf["metrics"]["val"]
    epochs1 = np.arange(1, ta1.shape[1] + 1)

    # Group-inverse
    p2 = ("experiment_results/"
          "experiment_f0c5353ab5d64aecb188e97426afb8dc_proc_17029/"
          "experiment_data.npy")
    gif = np.load(p2, allow_pickle=True).item()["group_inverse_frequency_reweighting"]["synthetic"]
    lrs2 = gif["lrs"]
    ta2, va2 = gif["metrics"]["train"], gif["metrics"]["val"]
    epochs2 = np.arange(1, ta2.shape[1] + 1)

    # Best LR for each
    idx1 = np.argmax(lrs1)
    idx2 = np.argmax(lrs2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Input-Feature
    axs[0].plot(epochs1, ta1[idx1], label="train")
    axs[0].plot(epochs1, va1[idx1], "--", label="val")
    axs[0].set_title(f"Input-Feature Reweighting\n(lr={lrs1[idx1]:.0e})")
    # Group-Inverse
    axs[1].plot(epochs2, ta2[idx2], label="train")
    axs[1].plot(epochs2, va2[idx2], "--", label="val")
    axs[1].set_title(f"Group-Inverse Reweighting\n(lr={lrs2[idx2]:.0e})")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=10)
        despine(ax)
    fig.suptitle("Synthetic: Reweighting Strategy Comparison", y=0.95)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig("figures/fig3_reweight_comparison.png")
    plt.close(fig)
except Exception as e:
    print("Error in fig3:", e)

# 4) Representation-Based Reweighting: Test Accuracy Bar
try:
    path = ("experiment_results/"
            "experiment_f8323db602414c9bbf09ab14c0ca8a84_proc_17030/"
            "experiment_data.npy")
    rep = np.load(path, allow_pickle=True).item()["representation_cluster_reweighting"]["synthetic"]
    accs = (rep["predictions"] == rep["ground_truth"][None, :]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(accs)), accs, color="skyblue")
    ax.set_xticks(np.arange(len(accs)))
    ax.set_xticklabels([f"Run {i+1}" for i in range(len(accs))])
    ax.set_xlabel("Run")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Representation-Based Reweighting Test Accuracy")
    despine(ax)
    fig.tight_layout()
    fig.savefig("figures/fig4_repr_test_accuracy.png")
    plt.close(fig)
except Exception as e:
    print("Error in fig4:", e)

# 5) Ablation Comparison: Final Validation WG Accuracy Bar
try:
    # No-cluster
    p_nc = ("experiment_results/"
            "experiment_d8cfcdf8be9e46229e44a653b9b38d14_proc_17030/"
            "experiment_data.npy")
    nc = np.load(p_nc, allow_pickle=True).item()["NO_CLUSTER_REWEIGHTING"]["synthetic"]
    val_nc = nc["metrics"]["val"][:, -1]

    # Linear classifier
    p_lc = ("experiment_results/"
            "experiment_1c2c85e9136d4bc79e9d19e2cc69aa0a_proc_17031/"
            "experiment_data.npy")
    lc = np.load(p_lc, allow_pickle=True).item()["linear_classifier"]["synthetic"]
    val_lc = lc["metrics"]["val"][:, -1]

    fig, ax = plt.subplots(figsize=(6,4))
    runs = np.arange(len(val_nc)+len(val_lc))
    bars = np.concatenate([val_nc, val_lc])
    labels = [f"NC Run{i+1}" for i in range(len(val_nc))] + \
             [f"LC Run{i+1}" for i in range(len(val_lc))]
    ax.bar(runs, bars, color=["salmon"]*len(val_nc) + ["skyblue"]*len(val_lc))
    ax.set_xticks(runs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Validation Worst-Group Accuracy")
    ax.set_title("Ablation: No-Cluster vs Linear Classifier")
    despine(ax)
    fig.tight_layout()
    fig.savefig("figures/fig5_ablation_bar.png")
    plt.close(fig)
except Exception as e:
    print("Error in fig5:", e)

# 6) Weight Decay Variation: Final Validation WG Accuracy Bar
try:
    path = ("experiment_results/"
            "experiment_c5fe4942dd39487d8fe7be4923a04b18_proc_17031/"
            "experiment_data.npy")
    wd = np.load(path, allow_pickle=True).item()["weight_decay_variation"]["synthetic"]
    wds = wd["weight_decays"]
    val_wd = wd["metrics"]["val"][:, -1]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([f"{w:.0e}" for w in wds], val_wd, color="mediumpurple")
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel("Validation Worst-Group Accuracy")
    ax.set_title("Synthetic: Weight Decay Variation")
    despine(ax)
    fig.tight_layout()
    fig.savefig("figures/fig6_weight_decay_bar.png")
    plt.close(fig)
except Exception as e:
    print("Error in fig6:", e)

# 7) Feature Normalization Ablation: Final Validation WG Accuracy Bar
try:
    path = ("experiment_results/"
            "experiment_6769e710853c4056871ba296f9aad9ce_proc_17029/"
            "experiment_data.npy")
    fn = np.load(path, allow_pickle=True).item()["NO_FEATURE_NORMALIZATION"]
    val_no = fn["synthetic_no_norm"]["metrics"]["val"][:, -1]
    val_wi = fn["synthetic_with_norm"]["metrics"]["val"][:, -1]
    lr_labels = [f"{lr:.0e}" for lr in [1e-4, 1e-3, 1e-2]]

    x = np.arange(len(lr_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x - width/2, val_no, width, label="No Norm", color="gray")
    ax.bar(x + width/2, val_wi, width, label="With Norm", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Worst-Group Accuracy")
    ax.set_title("Feature Normalization Ablation")
    ax.legend()
    despine(ax)
    fig.tight_layout()
    fig.savefig("figures/fig7_feature_norm_bar.png")
    plt.close(fig)
except Exception as e:
    print("Error in fig7:", e)