import os
import numpy as np
import matplotlib.pyplot as plt

# Create figures directory
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Global styling
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False
})

def disp(name):
    return name.replace("_", " ")

# 1) Baseline: 2×3 grid of accuracy and loss for three batch‐size configs
try:
    path = os.path.join(
        os.getcwd(),
        "experiment_results/experiment_a21677197dda42a9aa2af18492cef91e_proc_2565128/experiment_data.npy"
    )
    base = np.load(path, allow_pickle=True).item()["batch_size"]
    configs = ["ai_bs_16_user_bs_16", "ai_bs_32_user_bs_32", "ai_bs_64_user_bs_64"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), dpi=300)
    for col, cfg in enumerate(configs):
        m = base[cfg]["metrics"]
        x = np.arange(1, len(m["train"]) + 1)
        ax = axs[0, col]
        ax.plot(x, m["train"], label="Train accuracy")
        ax.plot(x, m["val"], label="Validation accuracy")
        ax.set_title(f"{disp(cfg)} accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    for col, cfg in enumerate(configs):
        l = base[cfg]["losses"]
        x = np.arange(1, len(l["train"]) + 1)
        ax = axs[1, col]
        ax.plot(x, l["train"], label="Train loss")
        ax.plot(x, l["val"], label="Validation loss")
        ax.set_title(f"{disp(cfg)} loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("Synthetic Binary Classification: Baseline Metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "baseline_metrics.png"))
    plt.close(fig)
except Exception as e:
    print("Error in baseline plotting:", e)

# 2) Label Input Ablation: Soft vs Hard (1×2)
try:
    path = os.path.join(
        os.getcwd(),
        "experiment_results/experiment_acfdb6e7d5a64d37808a07983dbd34e6_proc_2577153/experiment_data.npy"
    )
    lab = np.load(path, allow_pickle=True).item()
    types = list(lab.keys())
    cfg = list(set(lab[types[0]].keys()) & set(lab[types[1]].keys()))[0]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    # Accuracy
    for t in types:
        m = lab[t][cfg]["metrics"]
        axs[0].plot(m["train"], label=f"{disp(t)} train")
        axs[0].plot(m["val"], "--", label=f"{disp(t)} validation")
    axs[0].set_title(f"Label Ablation ({disp(cfg)}) – Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    # Loss
    for t in types:
        l = lab[t][cfg]["losses"]
        axs[1].plot(l["train"], label=f"{disp(t)} train")
        axs[1].plot(l["val"], "--", label=f"{disp(t)} validation")
    axs[1].set_title(f"Label Ablation ({disp(cfg)}) – Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "label_input_ablation.png"))
    plt.close(fig)
except Exception as e:
    print("Error in label ablation plotting:", e)

# 3) Teacher‐Feature Removal Ablation (1×2)
try:
    path = os.path.join(
        os.getcwd(),
        "experiment_results/experiment_33a5a1e460e544bea66c611c4ce48198_proc_2577154/experiment_data.npy"
    )
    tf = np.load(path, allow_pickle=True).item()["teacher_feature_removal"]
    scenarios = list(tf.keys())
    cfg = list(set(tf[scenarios[0]].keys()) & set(tf[scenarios[1]].keys()))[0]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    # Accuracy
    for s in scenarios:
        m = tf[s][cfg]["metrics"]
        axs[0].plot(m["train"], label=f"{disp(s)} train")
        axs[0].plot(m["val"], "--", label=f"{disp(s)} validation")
    axs[0].set_title(f"Teacher Feature Ablation ({disp(cfg)}) – Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    # Loss
    for s in scenarios:
        l = tf[s][cfg]["losses"]
        axs[1].plot(l["train"], label=f"{disp(s)} train")
        axs[1].plot(l["val"], "--", label=f"{disp(s)} validation")
    axs[1].set_title(f"Teacher Feature Ablation ({disp(cfg)}) – Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "teacher_feature_ablation.png"))
    plt.close(fig)
except Exception as e:
    print("Error in teacher‐feature ablation plotting:", e)

# 4) Confidence‐Filtered Pseudo‐Labeling Summary (1×3)
try:
    path = os.path.join(
        os.getcwd(),
        "experiment_results/experiment_9a98c8acd1524dc2b0ada48aab8154d9_proc_2577154/experiment_data.npy"
    )
    cf = np.load(path, allow_pickle=True).item()
    data = cf.get("confidence_filter", cf)
    keys = sorted(data.keys(), key=lambda k: float(k.split("_")[-1]))
    thr = [k.split("_")[-1] for k in keys]
    epochs = np.arange(1, len(data[keys[0]]["metrics"]["train"]) + 1)
    tacc = [(data[k]["predictions"] == data[k]["ground_truth"]).mean() for k in keys]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    # Accuracy curves
    for k, t in zip(keys, thr):
        m = data[k]["metrics"]
        axs[0].plot(epochs, m["train"], label=f"Train thr={t}")
        axs[0].plot(epochs, m["val"], "--", label=f"Val thr={t}")
    axs[0].set_title("Confidence Ablation – Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(fontsize="small", ncol=2)
    # Loss curves
    for k, t in zip(keys, thr):
        l = data[k]["losses"]
        axs[1].plot(epochs, l["train"], label=f"Train thr={t}")
        axs[1].plot(epochs, l["val"], "--", label=f"Val thr={t}")
    axs[1].set_title("Confidence Ablation – Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(fontsize="small", ncol=2)
    # Test accuracy bar
    axs[2].bar(range(len(thr)), tacc, color="skyblue")
    axs[2].set_xticks(range(len(thr)))
    axs[2].set_xticklabels(thr)
    axs[2].set_title("Confidence Ablation – Test Accuracy")
    axs[2].set_xlabel("Threshold")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_ylim(0, 1.05)
    for i, v in enumerate(tacc):
        axs[2].text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig.suptitle("Synthetic Classification: Confidence‐Filtered Pseudo‐Labeling")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "confidence_ablation_summary.png"))
    plt.close(fig)
except Exception as e:
    print("Error in confidence ablation plotting:", e)

# 5) Class Imbalance Ablation: one figure with three subplots
try:
    path = os.path.join(
        os.getcwd(),
        "experiment_results/experiment_ef723dac565c414b8b2fe812bb03a019_proc_2577154/experiment_data.npy"
    )
    ci = np.load(path, allow_pickle=True).item().get("class_imbalance", {})
    ratios = list(ci.keys())
    fig, axs = plt.subplots(1, len(ratios), figsize=(6 * len(ratios), 5), dpi=300)
    if len(ratios) == 1:
        axs = [axs]
    for ax, r in zip(axs, ratios):
        d = ci[r]
        keys = sorted(d.keys())
        accs = [(d[k]["predictions"] == d[k]["ground_truth"]).mean() for k in keys]
        ax.bar(range(len(keys)), accs, color="coral")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([disp(k) for k in keys], rotation=45, ha="right")
        ax.set_title(f"Class Imbalance {r.replace('_',':')}")
        ax.set_xlabel("Batch settings")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
    fig.suptitle("Synthetic Classification: Class Imbalance Ablation")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "class_imbalance_ablation.png"))
    plt.close(fig)
except Exception as e:
    print("Error in class imbalance plotting:", e)

# 6) Activation Function Ablation (Appendix): 1×3
try:
    path = os.path.join(
        os.getcwd(),
        "experiment_results/experiment_f62911705c734b32a40898c9fdcd82de_proc_2577153/experiment_data.npy"
    )
    act = np.load(path, allow_pickle=True).item()
    synth = {a: act[a].get("synthetic", act[a]) for a in act}
    acts = list(synth.keys())
    ep = np.arange(1, len(synth[acts[0]]["metrics"]["train"]) + 1)
    tacc = {a: (synth[a]["predictions"] == synth[a]["ground_truth"]).mean()
            for a in acts}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    # User accuracy
    for a in acts:
        m = synth[a]["metrics"]
        axs[0].plot(ep, m["train"], label=f"{disp(a)} train")
        axs[0].plot(ep, m["val"], "--", label=f"{disp(a)} validation")
    axs[0].set_title("User Accuracy by Activation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(fontsize="small", ncol=2)
    # User loss
    for a in acts:
        l = synth[a]["losses"]
        axs[1].plot(ep, l["train"], label=f"{disp(a)} train")
        axs[1].plot(ep, l["val"], "--", label=f"{disp(a)} validation")
    axs[1].set_title("User Loss by Activation")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(fontsize="small", ncol=2)
    # Test accuracy bar
    for i, a in enumerate(acts):
        axs[2].bar(i, tacc[a], color="seagreen")
        axs[2].text(i, tacc[a] + 0.005, f"{tacc[a]:.2f}", ha="center")
    axs[2].set_xticks(range(len(acts)))
    axs[2].set_xticklabels([disp(a) for a in acts], rotation=45, ha="right")
    axs[2].set_title("Activation Ablation – Test Accuracy")
    axs[2].set_xlabel("Activation")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_ylim(0, 1.05)
    fig.suptitle("Synthetic Classification: Activation Function Ablation (Appendix)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "activation_ablation_appendix.png"))
    plt.close(fig)
except Exception as e:
    print("Error in activation ablation plotting:", e)