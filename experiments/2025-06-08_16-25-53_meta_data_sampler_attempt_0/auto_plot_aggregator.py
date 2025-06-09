import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 14})

# 1) Synthetic baseline: Loss and Spearman in one figure
try:
    syn_path = os.path.join(
        "experiment_results",
        "experiment_57ddd97f0ac5481ab77c3607a38a5106_proc_234628",
        "experiment_data.npy",
    )
    syn = np.load(syn_path, allow_pickle=True).item()[
        "hyperparam_tuning_type_1"
    ]["synthetic"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    # Loss curves
    for p, tr, vl in zip(syn["param_values"], syn["losses"]["train"], syn["losses"]["val"]):
        x = np.arange(1, len(tr) + 1)
        axs[0].plot(x, tr, label=f"{p} epochs train")
        axs[0].plot(x, vl, "--", label=f"{p} epochs val")
    axs[0].set_title("Synthetic Training versus Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="best")
    axs[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Spearman correlation
    for p, corr in zip(syn["param_values"], syn["correlations"]):
        x = np.arange(1, len(corr) + 1)
        axs[1].plot(x, corr, marker="o", label=f"{p} epochs")
    axs[1].set_title("Synthetic Spearman Correlation")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Spearman ρ")
    axs[1].legend(loc="best")
    axs[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Synthetic Dataset Diagnostics", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/synthetic_summary.png")
    plt.close(fig)
except Exception as e:
    print(f"[Synthetic] Error: {e}")

# 2) Classification: Validation Loss and Accuracy (3 datasets)
try:
    class_path = os.path.join(
        "experiment_results",
        "experiment_00e6b1fa6e634ab2a71a23d999f23588_proc_247309",
        "experiment_data.npy",
    )
    cls = np.load(class_path, allow_pickle=True).item()
    ds_list = list(cls.keys())  # ['ag_news','yelp','dbpedia']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    for i, ds in enumerate(ds_list):
        stats = cls[ds]
        epochs = np.arange(1, len(stats["val_loss"]) + 1)

        # Validation loss
        ax = axes[0, i]
        ax.plot(epochs, stats["val_loss"], marker="o", color=f"C{i}")
        ax.set_title(f"{ds} Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Validation accuracy
        ax = axes[1, i]
        ax.plot(epochs, stats["val_acc"], marker="s", color=f"C{i}")
        ax.set_title(f"{ds} Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Text Classification Validation Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/classif_validation_metrics.png")
    plt.close(fig)
except Exception as e:
    print(f"[Classification Val] Error: {e}")

# 3) Classification: Meta Learning Dynamics (Spearman and N meta)
try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    for i, ds in enumerate(ds_list):
        stats = cls[ds]

        # Spearman correlation
        corr = np.array(stats.get("corrs", []))
        ax = axes[0, i]
        if corr.size > 0:
            steps = np.arange(1, corr.size + 1)
            ax.plot(steps, corr, marker="^", color=f"C{i}")
        ax.set_title(f"{ds} Spearman Correlation")
        ax.set_xlabel("Meta-update Step")
        ax.set_ylabel("Spearman ρ")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # N meta history
        nmeta = np.array(stats.get("N_meta_history", []))
        ax = axes[1, i]
        if nmeta.size > 0:
            steps = np.arange(1, nmeta.size + 1)
            ax.plot(steps, nmeta, marker="d", color=f"C{i}")
        ax.set_title(f"{ds} N meta History")
        ax.set_xlabel("Meta-update Step")
        ax.set_ylabel("N meta")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Text Classification Meta Learning Dynamics", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/classif_meta_dynamics.png")
    plt.close(fig)
except Exception as e:
    print(f"[Classification Meta] Error: {e}")

# 4) Ablation Weight Softmax Normalization (Spearman vs N meta)
try:
    w_path = os.path.join(
        "experiment_results",
        "experiment_686042838d6c4e4c9640e7f3a2576b00_proc_282857",
        "experiment_data.npy",
    )
    wnorm = np.load(w_path, allow_pickle=True).item()
    modes = list(wnorm.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    # Spearman correlation
    for m in modes:
        for ds, st in wnorm[m].items():
            corr = np.array(st.get("corrs", []))
            if corr.size > 0:
                axes[0].plot(
                    np.arange(1, corr.size + 1), corr, label=f"{m} ‑ {ds}"
                )
    axes[0].set_title("Ablation Spearman Correlation")
    axes[0].set_xlabel("Meta-update Step")
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend(fontsize=10)
    axes[0].grid(linestyle="--", alpha=0.5)

    # N meta history
    for m in modes:
        for ds, st in wnorm[m].items():
            nmeta = np.array(st.get("N_meta_history", []))
            if nmeta.size > 0:
                axes[1].plot(
                    np.arange(1, nmeta.size + 1), nmeta, label=f"{m} ‑ {ds}"
                )
    axes[1].set_title("Ablation N meta History")
    axes[1].set_xlabel("Meta-update Step")
    axes[1].set_ylabel("N meta")
    axes[1].legend(fontsize=10)
    axes[1].grid(linestyle="--", alpha=0.5)

    fig.suptitle("Weight Softmax Normalization Ablation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/ablate_weight_norm_summary.png")
    plt.close(fig)
except Exception as e:
    print(f"[Weight Norm Ablation] Error: {e}")

# 5) Ablation Label Noise Robustness on AG News (4×3 grid)
try:
    ln_path = os.path.join(
        "experiment_results",
        "experiment_9077c81eb095435482f75841a7eaca16_proc_282856",
        "experiment_data.npy",
    )
    ln = np.load(ln_path, allow_pickle=True).item()[
        "Ablate_Label_Noise_Robustness"
    ]["ag_news"]
    noises = sorted(ln.keys(), key=lambda x: int(x.replace("%", "")))

    fig, axs = plt.subplots(4, 3, figsize=(18, 12), dpi=300)
    for j, n in enumerate(noises):
        # Val accuracy
        acc = ln[n]["metrics"]["val"]
        x = np.arange(1, len(acc) + 1)
        axs[0, j].plot(x, acc, marker="o")
        axs[0, j].set_title(f"{n} noise – Val Accuracy")
        axs[0, j].set_xlabel("Epoch")
        axs[0, j].set_ylabel("Accuracy")
        axs[0, j].grid(axis="y", linestyle="--", alpha=0.5)

        # Val loss
        loss = ln[n]["losses"]["val"]
        axs[1, j].plot(x, loss, marker="s", color="C1")
        axs[1, j].set_title(f"{n} noise – Val Loss")
        axs[1, j].set_xlabel("Epoch")
        axs[1, j].set_ylabel("Loss")
        axs[1, j].grid(axis="y", linestyle="--", alpha=0.5)

        # Spearman correlation
        corr = np.array(ln[n].get("corrs", []))
        if corr.size > 0:
            xs = np.arange(1, corr.size + 1)
            axs[2, j].plot(xs, corr, marker="^", color="C2")
        axs[2, j].set_title(f"{n} noise – Spearman ρ")
        axs[2, j].set_xlabel("Meta-update Step")
        axs[2, j].set_ylabel("Spearman ρ")
        axs[2, j].grid(axis="y", linestyle="--", alpha=0.5)

        # N meta history
        nmeta = np.array(ln[n].get("N_meta_history", []))
        if nmeta.size > 0:
            xs = np.arange(1, nmeta.size + 1)
            axs[3, j].plot(xs, nmeta, marker="d", color="C3")
        axs[3, j].set_title(f"{n} noise – N meta")
        axs[3, j].set_xlabel("Meta-update Step")
        axs[3, j].set_ylabel("N meta")
        axs[3, j].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Label Noise Robustness Ablation on AG News", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/ablate_label_noise_agnews.png")
    plt.close(fig)
except Exception as e:
    print(f"[Label Noise Ablation] Error: {e}")

# 6) Ablation Representation Norm Feature Removed (2×3 grid)
try:
    rn_path = os.path.join(
        "experiment_results",
        "experiment_041aa9f9d0fa4be6a07c951cd1ccc8c4_proc_282856",
        "experiment_data.npy",
    )
    rn = np.load(rn_path, allow_pickle=True).item()[
        "Ablate_Representation_Norm_Feature"
    ]
    ds_rn = list(rn.keys())  # ['yelp','ag_news','dbpedia']

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), dpi=300)
    for j, ds in enumerate(ds_rn):
        x = np.arange(1, len(rn[ds]["metrics"]["val"]) + 1)
        # Val accuracy
        axs[0, j].plot(x, rn[ds]["metrics"]["val"], marker="o")
        axs[0, j].set_title(f"{ds} – Val Accuracy without Rep Norm")
        axs[0, j].set_xlabel("Epoch")
        axs[0, j].set_ylabel("Accuracy")
        axs[0, j].grid(axis="y", linestyle="--", alpha=0.5)
        # Val loss
        axs[1, j].plot(x, rn[ds]["losses"]["val"], marker="s", color="C1")
        axs[1, j].set_title(f"{ds} – Val Loss without Rep Norm")
        axs[1, j].set_xlabel("Epoch")
        axs[1, j].set_ylabel("Loss")
        axs[1, j].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Ablation: Representation Norm Feature Removed", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/ablate_no_repnorm.png")
    plt.close(fig)
except Exception as e:
    print(f"[Representation Norm Ablation] Error: {e}")