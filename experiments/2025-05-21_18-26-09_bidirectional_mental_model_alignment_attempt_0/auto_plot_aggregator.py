import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False
})

os.makedirs("figures", exist_ok=True)

try:
    lp_path = ("experiment_results/"
               "experiment_9529c89532744f2795767a760cae81c1_proc_4081279/"
               "experiment_data.npy")
    lp_data = np.load(lp_path, allow_pickle=True).item()
    datasets = list(lp_data['baseline'].keys())
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, ds in enumerate(datasets):
        epochs = np.arange(1, len(lp_data['baseline'][ds]['metrics']['train']) + 1)
        axs[0, i].plot(epochs, lp_data['baseline'][ds]['metrics']['val'], marker='o', label='Baseline')
        axs[0, i].plot(epochs, lp_data['linear_probe'][ds]['metrics']['val'], marker='^', label='Linear Probe')
        axs[0, i].set_title(f"{ds.replace('_',' ').title()} Accuracy")
        axs[0, i].set_xlabel("Epoch")
        axs[0, i].set_ylabel("Validation Accuracy")
        axs[0, i].legend()
        axs[1, i].plot(epochs, lp_data['baseline'][ds]['losses']['val'], marker='o', label='Baseline')
        axs[1, i].plot(epochs, lp_data['linear_probe'][ds]['losses']['val'], marker='^', label='Linear Probe')
        axs[1, i].set_title(f"{ds.replace('_',' ').title()} Loss")
        axs[1, i].set_xlabel("Epoch")
        axs[1, i].set_ylabel("Validation Loss")
        axs[1, i].legend()
    fig.suptitle("Baseline vs Linear Probe Ablation\nValidation Accuracy (top) and Loss (bottom) Across Datasets", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/fig1_linear_probe_vs_baseline.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in MAIN FIGURE 1: {e}")

try:
    drop_path = ("experiment_results/"
                 "experiment_c01d46ac1e9c48da88cc5010da1fc00e_proc_4081279/"
                 "experiment_data.npy")
    drop_data = np.load(drop_path, allow_pickle=True).item()["mlp_dropout_rate_ablation"]
    drop_keys = sorted(drop_data.keys(), key=lambda k: float(k.split("_")[1]))
    ds = "dbpedia_14"
    epochs = np.arange(1, len(drop_data[drop_keys[0]][ds]["losses"]["val"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for dk in drop_keys:
        p = dk.split("_")[1]
        axs[0].plot(epochs, drop_data[dk][ds]["losses"]["val"], label=f"dropout {p}%")
        axs[1].plot(epochs, drop_data[dk][ds]["alignments"]["val"], label=f"dropout {p}%")
        axs[2].plot(epochs, drop_data[dk][ds]["mai"], label=f"dropout {p}%")
    axs[0].set_title("Validation Loss\nDbpedia 14")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].set_title("Validation Alignment\nDbpedia 14")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Alignment")
    axs[1].legend()
    axs[2].set_title("Validation MAI\nDbpedia 14")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("MAI")
    axs[2].legend()
    fig.suptitle("MLP Dropout Rate Ablation on Dbpedia 14\nValidation Metrics Across Dropout Rates", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/fig2_mlp_dropout_dbpedia14.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in MAIN FIGURE 2: {e}")

try:
    distil_path = ("experiment_results/"
                   "experiment_33b1f7cfac554a38be76dd24c08c5ae6_proc_4081280/"
                   "experiment_data.npy")
    distil_data = np.load(distil_path, allow_pickle=True).item()
    ds = "dbpedia_14"
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ab in sorted(distil_data.keys()):
        losses = distil_data[ab][ds]["losses"]["val"]
        align = distil_data[ab][ds]["alignments"]["val"]
        mai = distil_data[ab][ds]["mai"]
        epochs = np.arange(1, len(losses) + 1)
        axs[0].plot(epochs, losses, label=ab)
        axs[1].plot(epochs, align, label=ab)
        axs[2].plot(epochs, mai, label=ab)
    axs[0].set_title("Validation Loss\nDbpedia 14")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].set_title("Validation Alignment\nDbpedia 14")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Alignment")
    axs[1].legend()
    axs[2].set_title("Validation MAI\nDbpedia 14")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("MAI")
    axs[2].legend()
    fig.suptitle("DistilBERT Pre-training Ablation on Dbpedia 14\nPretrained vs Random Init", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/fig3_distilbert_pretraining_dbpedia14.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in MAIN FIGURE 3: {e}")

try:
    ar_path = ("experiment_results/"
               "experiment_e1adbc0569d4459b888faf7a298452ab_proc_4081280/"
               "experiment_data.npy")
    ar_data = np.load(ar_path, allow_pickle=True).item()
    lam_keys = sorted(ar_data.keys(), key=lambda x: float(x.split("_")[1]))
    ds = "ag_news"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for lam in lam_keys:
        losses = ar_data[lam][ds]["losses"]["val"]
        accs = ar_data[lam][ds]["metrics"]["val"]
        epochs = np.arange(1, len(losses) + 1)
        axs[0].plot(epochs, losses, label=lam)
        axs[1].plot(epochs, accs, label=lam)
    axs[0].set_title("Validation Loss\nAG News")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].set_title("Validation Accuracy\nAG News")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    fig.suptitle("Alignment Regularization Ablation\nLambda Sweep on AG News", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/fig4_alignment_reg_ag_news.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in MAIN FIGURE 4: {e}")

try:
    opt_path = ("experiment_results/"
                "experiment_8b1e5415f46b4fd4a24eeaf1ff52ebec_proc_4081279/"
                "experiment_data.npy")
    opt_data = np.load(opt_path, allow_pickle=True).item()
    optimizers = list(opt_data.keys())
    datasets = list(opt_data[optimizers[0]].keys())
    fig, axs = plt.subplots(1, len(datasets), figsize=(15, 4))
    for i, ds in enumerate(datasets):
        for opt in optimizers:
            lval = opt_data[opt][ds]["losses"]["val"]
            epochs = np.arange(1, len(lval) + 1)
            axs[i].plot(epochs, lval, label=f"{opt} val", linestyle='--')
        axs[i].set_title(f"{ds.replace('_',' ').title()}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Validation Loss")
        axs[i].legend()
    fig.suptitle("Optimizer Choice Ablation\nValidation Loss Across Datasets", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/fig5_optimizer_choice_loss.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in MAIN FIGURE 5: {e}")

try:
    td_path = ("experiment_results/"
               "experiment_ec7d598449024b57a51ff39af928139d_proc_4081278/"
               "experiment_data.npy")
    td_data = np.load(td_path, allow_pickle=True).item()
    drop_keys = sorted(td_data.keys(), key=lambda x: int(x.split("_")[-1]))
    dataset_names = list(td_data[drop_keys[0]].keys())
    ds0 = "ag_news"
    epochs0 = np.arange(1, len(td_data[drop_keys[0]][ds0]["metrics"]["val"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for key in drop_keys:
        rate = int(key.split("_")[-1])
        accs = td_data[key][ds0]["metrics"]["val"]
        axs[0].plot(epochs0, accs, marker='o', label=f"{rate}% dropout")
    axs[0].set_title("AG News\nValidation Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    rates = [int(k.split("_")[-1]) for k in drop_keys]
    for ds in dataset_names:
        mai_vals = [td_data[k][ds]["mai"][-1] for k in drop_keys]
        axs[1].plot(rates, mai_vals, marker='o', label=ds.replace('_',' ').title())
    axs[1].set_title("Final Epoch MAI\nvs Token Dropout Rate")
    axs[1].set_xlabel("Token Dropout Rate (%)")
    axs[1].set_ylabel("MAI")
    axs[1].legend()
    fig.suptitle("Token Dropout Ablation\nAG News Accuracy & MAI Trends", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/fig6_token_dropout.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in MAIN FIGURE 6: {e}")

try:
    synth_path = ("experiment_results/"
                  "experiment_7d04e00cb67f4c5b995391df64eb1749_proc_4007053/"
                  "experiment_data.npy")
    synth = np.load(synth_path, allow_pickle=True).item()["learning_rate"]["synthetic"]
    lrs = synth["lrs"]
    train_losses = synth["losses"]["train"]
    val_losses = synth["losses"]["val"]
    train_align = synth["metrics"]["train"]
    val_align = synth["metrics"]["val"]
    epochs = np.arange(1, len(train_losses[0]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for lr, tr in zip(lrs, train_losses):
        axs[0, 0].plot(epochs, tr, label=f"lr={lr}")
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    for lr, vl in zip(lrs, val_losses):
        axs[0, 1].plot(epochs, vl, label=f"lr={lr}")
    axs[0, 1].set_title("Validation Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    for lr, ta in zip(lrs, train_align):
        axs[1, 0].plot(epochs, ta, label=f"lr={lr}")
    axs[1, 0].set_title("Training Alignment")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Alignment")
    axs[1, 0].legend()
    for lr, va in zip(lrs, val_align):
        axs[1, 1].plot(epochs, va, label=f"lr={lr}")
    axs[1, 1].set_title("Validation Alignment")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Alignment")
    axs[1, 1].legend()
    fig.suptitle("Appendix A1: Synthetic Dataset Learning Rate Sweep\nLoss and Alignment Curves Across Learning Rates", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/app_synth_lr_sweep.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in APPENDIX FIGURE A1: {e}")

try:
    act_path = ("experiment_results/"
                "experiment_31b90be82e6447e180577bc2ef7e6136_proc_4081278/"
                "experiment_data.npy")
    act_data = np.load(act_path, allow_pickle=True).item()
    activations = list(act_data.keys())
    ds = "yelp_polarity"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for act in activations:
        losses = act_data[act][ds]["losses"]
        epochs = np.arange(1, len(losses["train"]) + 1)
        axs[0].plot(epochs, losses["train"], label=f"{act} train")
        axs[0].plot(epochs, losses["val"], linestyle="--", label=f"{act} val")
        axs[1].plot(epochs, act_data[act][ds]["mai"], marker='o', label=act)
    axs[0].set_title("Training vs Validation Loss\nYelp Polarity")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].set_title("MAI over Epochs\nYelp Polarity")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MAI")
    axs[1].legend()
    fig.suptitle("Appendix A2: Activation Function Ablation\nYelp Polarity Metrics by Activation", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/app_activation_yelp_polarity.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in APPENDIX FIGURE A2: {e}")

try:
    attn_path = ("experiment_results/"
                 "experiment_00b9d440cc414af5af98e9b2d351e02a_proc_4081279/"
                 "experiment_data.npy")
    attn_data = np.load(attn_path, allow_pickle=True).item()
    datasets = ["ag_news", "yelp_polarity", "dbpedia_14"]
    types = ["random", "importance"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, ds in enumerate(datasets):
        for t in types:
            data = attn_data.get(t, {}).get(ds, {})
            heads = data.get("head_counts", [])
            mais = data.get("mai", [])
            uniq = sorted(set(heads))
            final = []
            for h in uniq:
                idxs = [j for j, hh in enumerate(heads) if hh == h]
                final.append(mais[max(idxs)])
            axs[i].plot(uniq, final, marker='o', label=t.capitalize())
        axs[i].set_title(ds.replace('_',' ').title())
        axs[i].set_xlabel("Number of Heads")
        axs[i].set_ylabel("MAI")
        axs[i].legend()
    fig.suptitle("Appendix A3: Attention Head Ablation\nMAI vs Head Count Across Datasets", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/app_attention_MAI_vs_heads.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in APPENDIX FIGURE A3: {e}")

try:
    focal_path = ("experiment_results/"
                  "experiment_5423d385102944c4a70669fbe1229642_proc_4081278/"
                  "experiment_data.npy")
    focal = np.load(focal_path, allow_pickle=True).item()
    ab_keys = sorted(focal.keys(), key=lambda k: int(k.split("_")[-1]))
    datasets = list(focal[ab_keys[0]].keys())
    loss_avg, align_avg, mai_avg = {}, {}, {}
    for ab in ab_keys:
        ds_data = focal[ab]
        train_losses = np.array([ds_data[ds]["losses"]["train"] for ds in datasets])
        val_losses   = np.array([ds_data[ds]["losses"]["val"]   for ds in datasets])
        train_align  = np.array([ds_data[ds]["alignments"]["train"] for ds in datasets])
        val_align    = np.array([ds_data[ds]["alignments"]["val"]   for ds in datasets])
        train_mai    = np.array([ds_data[ds]["metrics"]["train"]    for ds in datasets])
        val_mai      = np.array([ds_data[ds]["metrics"]["val"]      for ds in datasets])
        loss_avg[ab]  = {"train": train_losses.mean(0), "val": val_losses.mean(0)}
        align_avg[ab] = {"train": train_align.mean(0),  "val": val_align.mean(0)}
        mai_avg[ab]   = {"train": train_mai.mean(0),    "val": val_mai.mean(0)}
    epochs = np.arange(1, len(loss_avg[ab_keys[0]]["train"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ab in ab_keys:
        gamma = ab.split("_")[-1]
        axs[0].plot(epochs, loss_avg[ab]["train"], "-o", label=f"γ={gamma} train")
        axs[0].plot(epochs, loss_avg[ab]["val"],   "--s", label=f"γ={gamma} val")
        axs[1].plot(epochs, align_avg[ab]["train"], "-o", label=f"γ={gamma} train")
        axs[1].plot(epochs, align_avg[ab]["val"],   "--s", label=f"γ={gamma} val")
        axs[2].plot(epochs, mai_avg[ab]["train"],   "-o", label=f"γ={gamma} train")
        axs[2].plot(epochs, mai_avg[ab]["val"],     "--s", label=f"γ={gamma} val")
    axs[0].set_title("Avg Loss vs Epoch\nAcross Datasets")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].set_title("Avg Alignment vs Epoch\nAcross Datasets")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Alignment")
    axs[1].legend()
    axs[2].set_title("Avg MAI vs Epoch\nAcross Datasets")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("MAI")
    axs[2].legend()
    fig.suptitle("Appendix A4: Focal Loss Ablation\nAveraged Metrics Across Datasets", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/app_focal_loss_avg_metrics.png")
    plt.close(fig)
except Exception as e:
    print(f"Error in APPENDIX FIGURE A4: {e}")