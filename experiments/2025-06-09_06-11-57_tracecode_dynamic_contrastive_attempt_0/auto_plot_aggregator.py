import os
import numpy as np
import matplotlib.pyplot as plt

# Plot defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 1) Synthetic Epoch Ablation
try:
    fp = ("experiment_results/"
          "experiment_52c025f14d09471881bea2a75607be65_proc_385044/"
          "experiment_data.npy")
    syn = np.load(fp, allow_pickle=True).item()["EPOCHS"]["synthetic"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for E in sorted(syn, key=int):
        ep = np.arange(1, len(syn[E]["losses"]["train"]) + 1)
        axs[0].plot(ep, syn[E]["losses"]["train"], label=f"Train E={E}")
        axs[0].plot(ep, syn[E]["losses"]["val"],   linestyle='--', label=f"Val E={E}")
        axs[1].plot(ep, syn[E]["metrics"]["train"], label=f"Train E={E}")
        axs[1].plot(ep, syn[E]["metrics"]["val"],   linestyle='--', label=f"Val E={E}")
    axs[0].set(title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss")
    axs[1].set(title="Retrieval Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
    for ax in axs:
        ax.legend(loc='best'); clean_axes(ax)
    fig.suptitle("Synthetic Dataset Epoch Ablation", y=0.94)
    fig.tight_layout(rect=[0,0.03,1,0.90])
    fig.savefig(f"{FIG_DIR}/synthetic_epoch_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Synthetic Epoch Ablation:", e)

# 2) Multi‐Dataset Synthetic Ablation
try:
    fp = ("experiment_results/"
          "experiment_5221f873ef9f49acba79f917e212757c_proc_400996/"
          "experiment_data.npy")
    md = np.load(fp, allow_pickle=True).item()["multi_dataset_synthetic_ablation"]
    names = list(md.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4 * len(names)))
    for i, nm in enumerate(names):
        blk = md[nm]["EPOCHS"]
        ax_acc, ax_loss = axes[i]
        for E in sorted(blk, key=int):
            ep = np.arange(1, len(blk[E]["metrics"]["train"]) + 1)
            ax_acc.plot(ep, blk[E]["metrics"]["train"], label=f"Train E={E}")
            ax_acc.plot(ep, blk[E]["metrics"]["val"],   linestyle='--', label=f"Val E={E}")
            ax_loss.plot(ep, blk[E]["losses"]["train"], label=f"Train E={E}")
            ax_loss.plot(ep, blk[E]["losses"]["val"],   linestyle='--', label=f"Val E={E}")
        ax_acc.set(title=f"{nm.capitalize()} Accuracy", xlabel="Epoch", ylabel="Accuracy")
        ax_loss.set(title=f"{nm.capitalize()} Loss", xlabel="Epoch", ylabel="Loss")
        for ax in (ax_acc, ax_loss):
            ax.legend(loc='best'); clean_axes(ax)
    fig.suptitle("Multi‐Dataset Synthetic Ablation", y=0.92)
    fig.tight_layout(rect=[0,0.03,1,0.90])
    fig.savefig(f"{FIG_DIR}/multi_dataset_synthetic_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Multi‐Dataset Ablation:", e)

# 3) Negative Sampling Hardness Ablation
try:
    fp = ("experiment_results/"
          "experiment_dcb9f657828c4f6dbf6d7dafa8238d07_proc_400997/"
          "experiment_data.npy")
    nd = np.load(fp, allow_pickle=True).item()
    types = ["random_negative", "hard_negative"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, t in enumerate(types):
        blk = nd[t]["synthetic"]
        ax_l, ax_a = axes[i]
        for E in sorted(blk, key=int):
            ep = np.arange(1, len(blk[E]["losses"]["train"]) + 1)
            ax_l.plot(ep, blk[E]["losses"]["train"], label=f"Train E={E}")
            ax_l.plot(ep, blk[E]["losses"]["val"],   linestyle='--', label=f"Val E={E}")
            ax_a.plot(ep, blk[E]["metrics"]["train"], label=f"Train E={E}")
            ax_a.plot(ep, blk[E]["metrics"]["val"],   linestyle='--', label=f"Val E={E}")
        ax_l.set(title=f"{t.replace('_',' ').capitalize()} Loss", xlabel="Epoch", ylabel="Loss")
        ax_a.set(title=f"{t.replace('_',' ').capitalize()} Accuracy", xlabel="Epoch", ylabel="Accuracy")
        for ax in (ax_l, ax_a):
            ax.legend(loc='best'); clean_axes(ax)
    fig.suptitle("Negative Sampling Hardness Ablation", y=0.93)
    fig.tight_layout(rect=[0,0.03,1,0.90])
    fig.savefig(f"{FIG_DIR}/negative_sampling_hardness.png")
    plt.close(fig)
except Exception as e:
    print("Error in Negative Sampling Hardness Ablation:", e)

# 4) Triplet Margin Ablation (3‐panel)
try:
    fp = ("experiment_results/"
          "experiment_2120288f76b5437ab67ae259f95ba70a_proc_400997/"
          "experiment_data.npy")
    tm = np.load(fp, allow_pickle=True).item()["triplet_margin_ablation"]["synthetic"]
    margins = sorted(tm.keys(), key=float)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for m in margins:
        ep = np.arange(1, len(tm[m]["losses"]["train"]) + 1)
        axs[0].plot(ep, tm[m]["losses"]["train"], label=f"Train m={m}")
        axs[0].plot(ep, tm[m]["losses"]["val"],   linestyle='--', label=f"Val m={m}")
        axs[1].plot(ep, tm[m]["metrics"]["train"], label=f"Train m={m}")
        axs[1].plot(ep, tm[m]["metrics"]["val"],   linestyle='--', label=f"Val m={m}")
    final_acc = [tm[m]["metrics"]["val"][-1] for m in margins]
    axs[2].plot(margins, final_acc, marker='o')
    axs[0].set(title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss")
    axs[1].set(title="Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
    axs[2].set(title="Final Val Accuracy vs Margin", xlabel="Margin", ylabel="Accuracy")
    for ax in axs:
        ax.legend(loc='best') if ax is not axs[2] else clean_axes(ax)
        clean_axes(ax)
    fig.suptitle("Triplet Margin Ablation", y=0.96)
    fig.tight_layout(rect=[0,0.03,1,0.92])
    fig.savefig(f"{FIG_DIR}/triplet_margin_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Triplet Margin Ablation:", e)

# 5) Distance Metric Ablation
try:
    fp = ("experiment_results/"
          "experiment_aad02a7804b94b168a8de324c50e464c_proc_400995/"
          "experiment_data.npy")
    dm = np.load(fp, allow_pickle=True).item()
    methods = ["euclidean", "cosine"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, met in enumerate(methods):
        blk = dm[met]["synthetic"]
        ax_l, ax_a = axes[i]
        for E in sorted(blk, key=int):
            ep = np.arange(1, len(blk[E]["losses"]["train"]) + 1)
            ax_l.plot(ep, blk[E]["losses"]["train"], label=f"Train E={E}")
            ax_l.plot(ep, blk[E]["losses"]["val"],   linestyle='--', label=f"Val E={E}")
            ax_a.plot(ep, blk[E]["metrics"]["train"], label=f"Train E={E}")
            ax_a.plot(ep, blk[E]["metrics"]["val"],   linestyle='--', label=f"Val E={E}")
        ax_l.set(title=f"{met.capitalize()} Loss", xlabel="Epoch", ylabel="Loss")
        ax_a.set(title=f"{met.capitalize()} Accuracy", xlabel="Epoch", ylabel="Accuracy")
        for ax in (ax_l, ax_a):
            ax.legend(loc='best'); clean_axes(ax)
    fig.suptitle("Distance Metric Ablation", y=0.93)
    fig.tight_layout(rect=[0,0.03,1,0.90])
    fig.savefig(f"{FIG_DIR}/distance_metric_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Distance Metric Ablation:", e)

# 6) Embedding Dimension Ablation (loss+accuracy)
try:
    fp = ("experiment_results/"
          "experiment_8009319c799940e694e8f42df978c1e1_proc_400997/"
          "experiment_data.npy")
    ed = np.load(fp, allow_pickle=True).item()["embed_hidden"]["synthetic"]
    dims = sorted(ed.keys(), key=int)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Loss
    for d in dims:
        ep = np.arange(1, len(ed[d]["losses"]["train"]) + 1)
        axs[0].plot(ep, ed[d]["losses"]["train"], label=f"Train dim={d}")
        axs[0].plot(ep, ed[d]["losses"]["val"],   linestyle='--', label=f"Val dim={d}")
    axs[0].set(title="Loss vs Epoch by Embedding Dimension", xlabel="Epoch", ylabel="Loss")
    clean_axes(axs[0])
    # Accuracy
    for d in dims:
        ep = np.arange(1, len(ed[d]["metrics"]["train"]) + 1)
        axs[1].plot(ep, ed[d]["metrics"]["train"], label=f"Train dim={d}")
        axs[1].plot(ep, ed[d]["metrics"]["val"],   linestyle='--', label=f"Val dim={d}")
    axs[1].set(title="Accuracy vs Epoch by Embedding Dimension", xlabel="Epoch", ylabel="Accuracy")
    clean_axes(axs[1])
    for ax in axs:
        ax.legend(loc='best')
    fig.suptitle("Embedding Dimension Ablation", y=0.94)
    fig.tight_layout(rect=[0,0.03,1,0.90])
    fig.savefig(f"{FIG_DIR}/embedding_dimension_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Embedding Dimension Ablation:", e)

# 7) Variable Renaming Invariance Ablation
try:
    fp = ("experiment_results/"
          "experiment_80f4639c872f417d835aa5c4e1f4eea1_proc_400996/"
          "experiment_data.npy")
    vr = np.load(fp, allow_pickle=True).item()["variable_renaming_invariance"]["synthetic"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for E in sorted(vr, key=int):
        ep = np.arange(1, len(vr[E]["losses"]["train"]) + 1)
        axs[0].plot(ep, vr[E]["losses"]["train"], label=f"Train E={E}")
        axs[0].plot(ep, vr[E]["losses"]["val"],   linestyle='--', label=f"Val E={E}")
        axs[1].plot(ep, vr[E]["metrics"]["train"],  label=f"Train E={E}")
        axs[1].plot(ep, vr[E]["metrics"]["val"],    linestyle='--', label=f"Val E={E}")
        axs[1].plot(ep, vr[E]["metrics"]["rename"], marker='o', linestyle=':', label=f"Rename E={E}")
    axs[0].set(title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss")
    axs[1].set(title="Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
    for ax in axs:
        ax.legend(loc='best'); clean_axes(ax)
    fig.suptitle("Variable Renaming Invariance Ablation", y=0.94)
    fig.tight_layout(rect=[0,0.03,1,0.90])
    fig.savefig(f"{FIG_DIR}/variable_renaming_invariance_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Variable Renaming Invariance Ablation:", e)

# 8) Architecture Ablation (BiLSTM vs UniLSTM vs CNN)
try:
    # load BiLSTM ablation
    fp1 = ("experiment_results/"
           "experiment_b78b574eaf7343a0833f9f2662cbc161_proc_400997/"
           "experiment_data.npy")
    bd = np.load(fp1, allow_pickle=True).item()["bidirectional_lstm_ablation"]["synthetic"]
    # load CNN encoder ablation
    fp2 = ("experiment_results/"
           "experiment_f1380c5e79ff46038a65c1799d529f94_proc_400995/"
           "experiment_data.npy")
    cnn = np.load(fp2, allow_pickle=True).item()["CNN_ENCODER_ABLATION"]["synthetic"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # top row: BiLSTM vs UniLSTM Loss and Accuracy
    ax_l1, ax_a1 = axes[0]
    for var, blk in bd.items():
        for E in sorted(blk, key=int):
            ep = np.arange(1, len(blk[E]["losses"]["train"]) + 1)
            ax_l1.plot(ep, blk[E]["losses"]["train"], label=f"{var} train E={E}")
            ax_l1.plot(ep, blk[E]["losses"]["val"],   linestyle='--', label=f"{var} val E={E}")
            ax_a1.plot(ep, blk[E]["metrics"]["train"], label=f"{var} train E={E}")
            ax_a1.plot(ep, blk[E]["metrics"]["val"],   linestyle='--', label=f"{var} val E={E}")
    ax_l1.set(title="LSTM Loss vs Epoch", xlabel="Epoch", ylabel="Loss")
    ax_a1.set(title="LSTM Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
    for ax in (ax_l1, ax_a1):
        ax.legend(loc='best'); clean_axes(ax)
    # bottom row: CNN encoder Loss and Accuracy
    ax_l2, ax_a2 = axes[1]
    for E in sorted(cnn, key=int):
        ep = np.arange(1, len(cnn[E]["losses"]["train"]) + 1)
        ax_l2.plot(ep, cnn[E]["losses"]["train"], label=f"CNN train E={E}")
        ax_l2.plot(ep, cnn[E]["losses"]["val"],   linestyle='--', label=f"CNN val E={E}")
        ax_a2.plot(ep, cnn[E]["metrics"]["train"], label=f"CNN train E={E}")
        ax_a2.plot(ep, cnn[E]["metrics"]["val"],   linestyle='--', label=f"CNN val E={E}")
    ax_l2.set(title="CNN Loss vs Epoch", xlabel="Epoch", ylabel="Loss")
    ax_a2.set(title="CNN Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy")
    for ax in (ax_l2, ax_a2):
        ax.legend(loc='best'); clean_axes(ax)
    fig.suptitle("Architecture Ablation: LSTM vs CNN", y=0.94)
    fig.tight_layout(rect=[0,0.03,1,0.90])
    fig.savefig(f"{FIG_DIR}/architecture_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Architecture Ablation:", e)

print("Finished generating figures in 'figures/'.")