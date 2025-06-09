import os
import numpy as np
import matplotlib.pyplot as plt

# Global plotting settings
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# 1) Baseline: MLP vs CNN loss at 10 epochs
try:
    bp = "experiment_results/experiment_ba6e0e5c2f904eeb9af493981feb0491_proc_3707563/experiment_data.npy"
    data = np.load(bp, allow_pickle=True).item()
    run10 = data["n_epochs"]["10"]["models"]
    mlp = run10["MLP"]["losses"]
    cnn = run10["CNN"]["losses"]
    e_mlp = np.arange(1, len(mlp["train"]) + 1)
    e_cnn = np.arange(1, len(cnn["train"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    ax1.plot(e_mlp, mlp["train"], label="Train Loss")
    ax1.plot(e_mlp, mlp["val"], "--", label="Val Loss")
    ax1.set_title("MLP Loss Curves (10 Epochs)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    style_ax(ax1)

    ax2.plot(e_cnn, cnn["train"], label="Train Loss")
    ax2.plot(e_cnn, cnn["val"], "--", label="Val Loss")
    ax2.set_title("CNN Loss Curves (10 Epochs)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    style_ax(ax2)

    fig.suptitle("MNIST Loss Curves at 10 Epochs (Baseline)", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/mnist_loss_mlp_cnn_10_epochs.png")
    plt.close(fig)
except Exception as e:
    print(f"[Baseline loss] {e}")

# 2) Baseline: CGR vs Epoch
try:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for n, run in sorted(data["n_epochs"].items(), key=lambda x: int(x[0])):
        cgr = run["cgr"]
        e = np.arange(1, len(cgr) + 1)
        ax.plot(e, cgr, marker="o", label=f"n_epochs={n}")
    ax.set_title("CGR vs Epoch on MNIST")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CGR")
    ax.legend(loc="best")
    style_ax(ax)
    plt.tight_layout()
    fig.savefig("figures/mnist_cgr_vs_epoch.png")
    plt.close(fig)
except Exception as e:
    print(f"[Baseline CGR] {e}")

# 3) Research: Discrimination score across text datasets
try:
    rp = "experiment_results/experiment_c6f4bdf859a041f698b3415320f73684_proc_3732491/experiment_data.npy"
    rdata = np.load(rp, allow_pickle=True).item()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for ds, d in rdata.items():
        if "discrimination_score" in d:
            dscr = d["discrimination_score"]
            e = np.arange(1, len(dscr) + 1)
            ax.plot(e, dscr, label=ds)
    ax.set_title("Discrimination Score Across Text Datasets")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discrimination Score")
    ax.legend(loc="best")
    style_ax(ax)
    plt.tight_layout()
    fig.savefig("figures/discrimination_score_text.png")
    plt.close(fig)
except Exception as e:
    print(f"[Discrimination] {e}")

# 4) Research: Final validation accuracy comparison
try:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    labels = [ds for ds in rdata if "discrimination_score" in rdata[ds]]
    models = list(rdata[labels[0]]["metrics"].keys())
    x = np.arange(len(labels))
    w = 0.2
    for i, m in enumerate(models):
        accs = [rdata[ds]["metrics"][m]["val_acc"][-1] for ds in labels]
        ax.bar(x + (i - 1)*w, accs, w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Validation Accuracy Across Text Datasets")
    ax.legend(loc="best")
    style_ax(ax)
    plt.tight_layout()
    fig.savefig("figures/final_val_accuracy_text.png")
    plt.close(fig)
except Exception as e:
    print(f"[Final val acc] {e}")

# 5) Ablation: Pooling mechanism (orig vs aug accuracy)
try:
    pp = "experiment_results/experiment_127dd01927f942b5a5cdb0915ca6d689_proc_3746534/experiment_data.npy"
    pdata = np.load(pp, allow_pickle=True).item()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    for variant, v in pdata.items():
        for eps_key, vals in v.items():
            eps = eps_key.split("_", 1)[1]
            o = vals["metrics"]["orig_acc"]
            a = vals["metrics"]["aug_acc"]
            e = np.arange(1, len(o) + 1)
            axes[0].plot(e, o, label=f"{variant} ε={eps}")
            axes[1].plot(e, a, "--", label=f"{variant} ε={eps}")
    axes[0].set_title("Original Test Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Augmented Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.legend(loc="best")
        style_ax(ax)
    fig.suptitle("Pooling Mechanism Ablation on MNIST", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/pooling_ablation.png")
    plt.close(fig)
except Exception as e:
    print(f"[Pooling] {e}")

# 6) Ablation: Training augmentation (loss + accuracy)
try:
    tp = "experiment_results/experiment_e98e188b67cf410f892a6941ee7b0767_proc_3746533/experiment_data.npy"
    tdata = np.load(tp, allow_pickle=True).item()
    cfgs = list(tdata.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    # Loss
    for c in cfgs:
        tr = tdata[c]["orig"]["losses"]["train"]
        vo = tdata[c]["orig"]["losses"]["val"]
        vr = tdata[c]["rot"]["losses"]["val"]
        e = np.arange(1, len(tr) + 1)
        axes[0].plot(e, tr, label=f"{c} train")
        axes[0].plot(e, vo, "--", label=f"{c} orig val")
        axes[0].plot(e, vr, ":", label=f"{c} rot val")
    axes[0].set_title("Loss Curves by Augmentation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    # Accuracy
    for c in cfgs:
        ao = tdata[c]["orig"]["metrics"]["acc"]
        ar = tdata[c]["rot"]["metrics"]["acc"]
        e = np.arange(1, len(ao) + 1)
        axes[1].plot(e, ao, label=f"{c} orig acc")
        axes[1].plot(e, ar, "--", label=f"{c} rot acc")
    axes[1].set_title("Accuracy Curves by Augmentation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.legend(loc="best")
        style_ax(ax)
    fig.suptitle("Training Augmentation Ablation on MNIST", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/augmentation_ablation.png")
    plt.close(fig)
except Exception as e:
    print(f"[Augmentation] {e}")

# 7) Ablation: Adversarial training (2x2 grid)
try:
    ap = "experiment_results/experiment_b56a9bf39dff48b2b02565ef9ff550e0_proc_3746535/experiment_data.npy"
    adata = np.load(ap, allow_pickle=True).item()["adversarial_training"]
    epses = sorted(float(k.split("_")[1]) for k in adata)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    for eps in epses:
        key = f"eps_{eps}"
        e = np.arange(1, len(adata[key]["clean"]["losses"]["train"]) + 1)
        # clean loss
        axs[0, 0].plot(e, adata[key]["clean"]["losses"]["train"], label=f"clean train ε={eps}")
        axs[0, 0].plot(e, adata[key]["clean"]["losses"]["val"], "--", label=f"clean val ε={eps}")
        # adv loss
        axs[0, 1].plot(e, adata[key]["adv"]["losses"]["train"], label=f"adv train ε={eps}")
        axs[0, 1].plot(e, adata[key]["adv"]["losses"]["val"], "--", label=f"adv val ε={eps}")
        # orig acc
        axs[1, 0].plot(e, adata[key]["clean"]["metrics"]["orig_acc"], label=f"clean orig ε={eps}")
        axs[1, 0].plot(e, adata[key]["adv"]["metrics"]["orig_acc"], "--", label=f"adv orig ε={eps}")
        # robust acc
        axs[1, 1].plot(e, adata[key]["clean"]["metrics"]["robust_acc"], label=f"clean rob ε={eps}")
        axs[1, 1].plot(e, adata[key]["adv"]["metrics"]["robust_acc"], "--", label=f"adv rob ε={eps}")
    titles = ["Clean Loss", "Adversarial Loss", "Original Accuracy", "Robust Accuracy"]
    for ax, t in zip(axs.flatten(), titles):
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.legend(loc="best")
        style_ax(ax)
    fig.suptitle("Adversarial Training Ablation on MNIST", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/adversarial_ablation.png")
    plt.close(fig)
except Exception as e:
    print(f"[Adversarial] {e}")

# 8) Ablation: Mixup augmentation
try:
    mp = "experiment_results/experiment_e5e08d2e79d14d55b3210d18b3f3942c_proc_3746534/experiment_data.npy"
    mdata = np.load(mp, allow_pickle=True).item()["mixup"]
    alphas = sorted(float(k.split("_")[1]) for k in mdata)
    e = np.arange(1, len(mdata[f"alpha_{alphas[0]}"]["losses"]["train"]) + 1)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for α in alphas:
        key = f"alpha_{α}"
        axL.plot(e, mdata[key]["losses"]["train"], label=f"train α={α}")
        axL.plot(e, mdata[key]["losses"]["val"], "--", label=f"val α={α}")
        axR.plot(e, mdata[key]["metrics"]["orig_acc"], label=f"orig α={α}")
        axR.plot(e, mdata[key]["metrics"]["aug_acc"], "--", label=f"aug α={α}")
    axL.set_title("Mixup: Training & Validation Loss")
    axL.set_xlabel("Epoch"); axL.set_ylabel("Loss")
    axR.set_title("Mixup: Original & Augmented Accuracy")
    axR.set_xlabel("Epoch"); axR.set_ylabel("Accuracy")
    for ax in (axL, axR):
        ax.legend(loc="best")
        style_ax(ax)
    fig.suptitle("Mixup Data Augmentation Ablation on MNIST", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/mixup_ablation.png")
    plt.close(fig)
except Exception as e:
    print(f"[Mixup] {e}")

# 9) Ablation: Learning rate scheduler
try:
    sp = "experiment_results/experiment_c5aeb1c14b1d40f2915957d842241438_proc_3746533/experiment_data.npy"
    sdata = np.load(sp, allow_pickle=True).item()["lr_scheduler"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for name, info in sdata.items():
        e = np.arange(1, len(info["losses"]["train"]) + 1)
        ax1.plot(e, info["losses"]["train"], label=f"{name} train")
        ax1.plot(e, info["losses"]["val"], "--", label=f"{name} val")
        ax2.plot(e, info["metrics"]["orig_acc"], label=f"{name} orig")
        ax2.plot(e, info["metrics"]["aug_acc"], "--", label=f"{name} aug")
    ax1.set_title("Scheduler: Train & Val Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.set_title("Scheduler: Orig & Aug Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    for ax in (ax1, ax2):
        ax.legend(loc="best")
        style_ax(ax)
    fig.suptitle("Learning Rate Scheduler Ablation on MNIST", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/lr_scheduler_ablation.png")
    plt.close(fig)
except Exception as e:
    print(f"[Scheduler] {e}")

# 10) Ablation: Weight decay
try:
    wp = "experiment_results/experiment_e35d4674ad2c447aad8b3749facfa5b4_proc_3746535/experiment_data.npy"
    wdata = np.load(wp, allow_pickle=True).item()["weight_decay"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for key, info in wdata.items():
        e = np.arange(1, len(info["losses"]["train"]) + 1)
        ax1.plot(e, info["losses"]["train"], label=key)
        ax1.plot(e, info["losses"]["val"], "--", label=key)
        ax2.plot(e, info["metrics"]["orig_acc"], label=f"{key} orig")
        ax2.plot(e, info["metrics"]["aug_acc"], "--", label=f"{key} aug")
    ax1.set_title("Weight Decay: Train & Val Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.set_title("Weight Decay: Orig & Aug Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    for ax in (ax1, ax2):
        ax.legend(loc="best"); style_ax(ax)
    fig.suptitle("Weight Decay Ablation on MNIST", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/weight_decay_ablation.png")
    plt.close(fig)
except Exception as e:
    print(f"[Weight Decay] {e}")

# 11) Ablation: Activation function
try:
    ap = "experiment_results/experiment_aa345f1e4fb440e382e95b129c2b8bab_proc_3746533/experiment_data.npy"
    act = np.load(ap, allow_pickle=True).item()["activation_function_ablation"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for name, info in act.items():
        e = np.arange(1, len(info["losses"]["train"]) + 1)
        ax1.plot(e, info["losses"]["train"], label=f"{name} train")
        ax1.plot(e, info["losses"]["val"], "--", label=f"{name} val")
        ax2.plot(e, info["metrics"]["orig_acc"], label=f"{name} orig")
        ax2.plot(e, info["metrics"]["aug_acc"], "--", label=f"{name} aug")
    ax1.set_title("Activation: Train & Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.set_title("Activation: Orig & Aug Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    for ax in (ax1, ax2):
        ax.legend(loc="best"); style_ax(ax)
    fig.suptitle("Activation Function Ablation on MNIST", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/activation_ablation.png")
    plt.close(fig)
except Exception as e:
    print(f"[Activation] {e}")