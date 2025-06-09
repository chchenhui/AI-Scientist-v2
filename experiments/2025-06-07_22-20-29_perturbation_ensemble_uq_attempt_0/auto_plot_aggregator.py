import os
import numpy as np
import matplotlib.pyplot as plt

# Professional style
plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300
})

os.makedirs("figures", exist_ok=True)

# Exact .npy paths
paths = {
    "baseline": "experiment_results/experiment_55e8fb977bb9465aad39c197cfd1a278_proc_144143/experiment_data.npy",
    "research": "experiment_results/experiment_2d32a622874442d19fddaa848b7f6367_proc_152740/experiment_data.npy",
    "head_only": "experiment_results/experiment_9b232c90eb8248edba7db412eecf37cb_proc_164440/experiment_data.npy",
    "no_pretrain": "experiment_results/experiment_1746817a2766428e8d23cb0f558d155c_proc_164442/experiment_data.npy",
    "no_dropout": "experiment_results/experiment_9d34db6142df40de818f797aa7734a09_proc_164442/experiment_data.npy",
    "positional": "experiment_results/experiment_74cd1572cf364cd9af47e780a766505d_proc_164440/experiment_data.npy",
    "token_type": "experiment_results/experiment_7856d3a5914e40dda4cca14396e9eb43_proc_164442/experiment_data.npy",
    "bias_removal": "experiment_results/experiment_7fb1bcfc326d4ecd9b86d2033aa9690f_proc_164440/experiment_data.npy",
    "depth": "experiment_results/experiment_1fbca80115dc4f94a62937611cfbdba3_proc_164442/experiment_data.npy"
}

pretty = {"sst2": "SST-2", "yelp_polarity": "Yelp Polarity", "imdb": "IMDb", "synthetic": "Synthetic"}

def load(key):
    try:
        return np.load(paths[key], allow_pickle=True).item()
    except:
        return {}

# 1) Main tasks: Loss and Detection AUC (2×3 grid)
res = load("research")
tasks = ["sst2", "yelp_polarity", "imdb"]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ds in enumerate(tasks):
    e = res.get(ds, {})
    # Loss curves
    ax = axes[0, i]
    tr = [x["loss"] for x in e.get("losses", {}).get("train", [])]
    vl = [x["loss"] for x in e.get("losses", {}).get("val", [])]
    ep = [x["epoch"] for x in e.get("losses", {}).get("train", [])]
    ax.plot(ep, tr, label="Train Loss")
    ax.plot(ep, vl, label="Validation Loss")
    ax.set_title(f"{pretty[ds]} Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    # Detection AUC
    ax = axes[1, i]
    det = e.get("metrics", {}).get("detection", [])
    ep = [d["epoch"] for d in det]
    av = [d["auc_vote"] for d in det]
    ak = [d["auc_kl"] for d in det]
    ax.plot(ep, av, label="Vote AUC")
    ax.plot(ep, ak, label="KL AUC")
    ax.set_title(f"{pretty[ds]} Detection AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
fig.tight_layout()
fig.suptitle("Main Tasks: Loss and Detection AUC", y=1.02)
fig.savefig("figures/figure1_main_tasks.png")
plt.close(fig)

# 2) Final Detection AUC Comparison
vote_final, kl_final = [], []
for ds in tasks:
    det = res.get(ds, {}).get("metrics", {}).get("detection", [])
    if det:
        vote_final.append(det[-1]["auc_vote"])
        kl_final.append(det[-1]["auc_kl"])
    else:
        vote_final.append(np.nan)
        kl_final.append(np.nan)
x = np.arange(len(tasks))
w = 0.35
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - w/2, vote_final, w, label="Vote AUC")
ax.bar(x + w/2, kl_final, w, label="KL AUC")
ax.set_xticks(x)
ax.set_xticklabels([pretty[t] for t in tasks])
ax.set_xlabel("Dataset")
ax.set_ylabel("Final Detection AUC")
ax.set_title("Final Detection AUC by Dataset")
ax.legend()
fig.tight_layout()
fig.savefig("figures/figure2_final_auc_comparison.png")
plt.close(fig)

# 3–5) Ablation: Final KL AUC per method for each dataset
methods = [
    ("Baseline", "baseline", None),
    ("Research", "research", None),
    ("Head Only", "head_only", "head_only"),
    ("No Pretraining", "no_pretrain", "No_Pretraining_RandomInit"),
    ("No Dropout", "no_dropout", "No_Dropout_Ablation"),
    ("Positional Embedding", "positional", "positional_embedding_ablation"),
    ("Token Type Embedding", "token_type", "token_type_embedding_ablation"),
    ("Bias Removal", "bias_removal", "BiasRemovalAblation"),
    ("Depth Full", "depth", "full_depth"),
    ("Depth Reduced", "depth", "reduced_depth")
]

for idx, ds in enumerate(tasks, start=3):
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = []
    labels = []
    for name, key, sub in methods:
        data = load(key)
        if sub:
            # depth has two sub-keys under depth
            if key == "depth":
                data = data.get(sub, {}).get(ds, {})
            else:
                data = data.get(sub, {})
        if key == "research":
            data = data.get(ds, {})
        det = data.get("metrics", {}).get("detection", [])
        kl = det[-1]["auc_kl"] if det else np.nan
        vals.append(kl)
        labels.append(name)
    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{pretty[ds]} Final KL AUC Across Methods")
    ax.set_ylabel("KL AUC")
    fig.tight_layout()
    fig.savefig(f"figures/figure{idx}_{ds}_ablation.png")
    plt.close(fig)

# 6) Appendix: Synthetic Data Curves
base = load("baseline").get("synthetic", {})
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# Loss
tr = [x["loss"] for x in base.get("losses", {}).get("train", [])]
vl = [x["loss"] for x in base.get("losses", {}).get("val", [])]
ep = [x["epoch"] for x in base.get("losses", {}).get("train", [])]
axes[0].plot(ep, tr, label="Train Loss")
axes[0].plot(ep, vl, label="Validation Loss")
axes[0].set_title("Synthetic Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
# AUC
t_auc = [d["auc"] for d in base.get("metrics", {}).get("train", [])]
v_auc = [d["auc"] for d in base.get("metrics", {}).get("val", [])]
ep_auc = [d["epoch"] for d in base.get("metrics", {}).get("train", [])]
axes[1].plot(ep_auc, t_auc, label="Train AUC")
axes[1].plot(ep_auc, v_auc, label="Validation AUC")
axes[1].set_title("Synthetic Classification AUC")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].legend()
fig.tight_layout()
fig.savefig("figures/figure6_synthetic_appendix.png")
plt.close(fig)

print("Generated 6 clean, aggregated figures in 'figures/'.")