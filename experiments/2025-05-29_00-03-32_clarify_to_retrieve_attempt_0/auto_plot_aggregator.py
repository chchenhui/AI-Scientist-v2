import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure high-resolution, professional style
plt.rcParams.update({'font.size': 14, 'figure.dpi': 300})
def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Create output folder
os.makedirs("figures", exist_ok=True)

# 1) Load all experiment data
# Baseline: synthetic XOR hyperparameter sweep
baseline_path = "experiment_results/experiment_3f33ddac87644ba2a1db2c8af87cae28_proc_2379938/experiment_data.npy"
try:
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
except:
    baseline_data = {}
# Research summary: QA datasets metrics
research_path = "experiment_results/experiment_b27ae5277c6c4eb8a82265830d4556fc_proc_2404625/experiment_data.npy"
try:
    research_data = np.load(research_path, allow_pickle=True).item()
except:
    research_data = {}
# Ablation: Clarification Turn Budget
budget_path = "experiment_results/experiment_4b37fcb2d2864e209d6b274cfaec7c2b_proc_2413995/experiment_data.npy"
try:
    budget_data = np.load(budget_path, allow_pickle=True).item()
except:
    budget_data = {}
# Ablation: Always-Ask Clarification
always_path = "experiment_results/experiment_755b3a65e2364e5dae6812bfce84a43f_proc_2413996/experiment_data.npy"
try:
    always_data = np.load(always_path, allow_pickle=True).item()
except:
    always_data = {}
# Ablation: Ambiguity Detection Noise
noise_det_path = "experiment_results/experiment_6ffd9ee8982e414fa8506aaace7fd458_proc_2413997/experiment_data.npy"
try:
    noise_det_data = np.load(noise_det_path, allow_pickle=True).item()
except:
    noise_det_data = {}
# Ablation: Post-Clarification Retrieval Noise
postclar_path = "experiment_results/experiment_2754fc14feec4023b4e77fd22018c8af_proc_2413995/experiment_data.npy"
try:
    postclar_data = np.load(postclar_path, allow_pickle=True).item()
except:
    postclar_data = {}
# Ablation: User Patience Dropout
patience_path = "experiment_results/experiment_49bac4e8c7614653b92022ddfc8c130f_proc_2413996/experiment_data.npy"
try:
    patience_data = np.load(patience_path, allow_pickle=True).item()
except:
    patience_data = {}
# Ablation: Clarification Question Format
qform_path = "experiment_results/experiment_24d9fc7c8e4a438ab94b1400730c0646_proc_2413995/experiment_data.npy"
try:
    qform_data = np.load(qform_path, allow_pickle=True).item()
except:
    qform_data = {}
# Ablation: Confidence-Threshold Clarification
confthr_path = "experiment_results/experiment_c8528ec2aefb45dc8543f472c459b32a_proc_2413997/experiment_data.npy"
try:
    confthr_data = np.load(confthr_path, allow_pickle=True).item()
except:
    confthr_data = {}
# Ablation: Multi-Passage Answer Fusion
mpaf_path = "experiment_results/experiment_da922064c5b74605b228d8a5f16d39f6_proc_2413996/experiment_data.npy"
try:
    mpaf_data = np.load(mpaf_path, allow_pickle=True).item()
except:
    mpaf_data = {}

# === Figure 1: Synthetic XOR Curves (Loss & CES) ===
try:
    syn = baseline_data.get("hidden_layer_size", {}).get("synthetic_xor", {})
    sizes = syn.get("sizes", [])
    losses_tr = syn.get("losses", {}).get("train", [])
    losses_val = syn.get("losses", {}).get("val", [])
    ces_tr = syn.get("metrics", {}).get("train", [])
    ces_val = syn.get("metrics", {}).get("val", [])
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    # Loss curves
    for sz, lt, lv in zip(sizes, losses_tr, losses_val):
        axs[0].plot(range(1, len(lt)+1), lt, label=f"Train (size={sz})")
        axs[0].plot(range(1, len(lv)+1), lv, "--", label=f"Val (size={sz})")
    axs[0].set_title("Synthetic XOR: Loss vs Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    clean_axes(axs[0])
    # CES curves
    for sz, ct, cv in zip(sizes, ces_tr, ces_val):
        axs[1].plot(range(1, len(ct)+1), ct, label=f"Train CES (size={sz})")
        axs[1].plot(range(1, len(cv)+1), cv, "--", label=f"Val CES (size={sz})")
    axs[1].set_title("Synthetic XOR: Calibration Error Score vs Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("CES")
    axs[1].legend()
    clean_axes(axs[1])
    fig.tight_layout()
    fig.savefig("figures/synthetic_xor_curves.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 1:", e)

# === Figure 2: Synthetic XOR Final CES by Hidden Size ===
try:
    final_ces = [m[-1] if m else 0 for m in syn.get("metrics", {}).get("val", [])]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([str(s) for s in sizes], final_ces, color="skyblue")
    ax.set_title("Synthetic XOR: Final Validation CES by Hidden Layer Size")
    ax.set_xlabel("Hidden Layer Size")
    ax.set_ylabel("CES")
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig("figures/synthetic_xor_final_CES.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 2:", e)

# === Figure 3: QA Datasets Summary (Acc, CES, Turns) ===
try:
    metrics = research_data.get("metrics", {})
    names = list(metrics.keys())
    baseline_acc = [metrics[n]["baseline_acc"] for n in names]
    clar_acc = [metrics[n]["clar_acc"] for n in names]
    ces_scores = [metrics[n]["CES"] for n in names]
    avg_turns = [metrics[n]["avg_turns"] for n in names]
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Acc
    x = np.arange(len(names))
    w = 0.35
    axs[0].bar(x - w/2, baseline_acc, w, label="Baseline")
    axs[0].bar(x + w/2, clar_acc, w, label="Clarified")
    axs[0].set_xticks(x); axs[0].set_xticklabels(names)
    axs[0].set_title("Baseline vs Clarification Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    clean_axes(axs[0])
    # CES
    axs[1].bar(names, ces_scores, color="orange")
    axs[1].set_title("Clarification Efficiency Score (CES)")
    axs[1].set_ylabel("CES")
    clean_axes(axs[1])
    # Avg turns
    axs[2].bar(names, avg_turns, color="gray")
    axs[2].set_title("Average Clarification Turns")
    axs[2].set_ylabel("Turns")
    clean_axes(axs[2])
    fig.tight_layout()
    fig.savefig("figures/qa_datasets_summary.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 3:", e)

# === Figure 4: Clarification Turn Budget Ablation ===
try:
    ctb = budget_data.get("clarification_turn_budget", {})
    ds = list(ctb.keys())
    # assuming each has ['metrics']['val'] -> list of dicts per budget
    val0 = ctb[ds[0]]["metrics"]["val"]
    budgets = [m["budget"] for m in val0]
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    # Accuracies
    for name in ds:
        vm = ctb[name]["metrics"]["val"]
        b_acc = [m["baseline_acc"] for m in vm]
        c_acc = [m["clar_acc"] for m in vm]
        axs[0].plot(budgets, b_acc, "o-", label=f"{name} Baseline")
        axs[0].plot(budgets, c_acc, "x--", label=f"{name} Clarified")
    axs[0].set_title("Budget vs Accuracy")
    axs[0].set_xlabel("Turn Budget"); axs[0].set_ylabel("Accuracy")
    axs[0].legend(); clean_axes(axs[0])
    # Avg turns
    for name in ds:
        vm = ctb[name]["metrics"]["val"]
        at = [m["avg_turns"] for m in vm]
        axs[1].plot(budgets, at, "o-", label=name)
    axs[1].set_title("Budget vs Avg Clarification Turns")
    axs[1].set_xlabel("Turn Budget"); axs[1].set_ylabel("Avg Turns")
    axs[1].legend(); clean_axes(axs[1])
    # CES
    for name in ds:
        vm = ctb[name]["metrics"]["val"]
        sc = [m["CES"] for m in vm]
        axs[2].plot(budgets, sc, "o-", label=name)
    axs[2].set_title("Budget vs CES")
    axs[2].set_xlabel("Turn Budget"); axs[2].set_ylabel("CES")
    axs[2].legend(); clean_axes(axs[2])
    fig.tight_layout()
    fig.savefig("figures/budget_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 4:", e)

# === Figure 5: Ambiguity Detection Noise Ablation ===
try:
    adn = noise_det_data.get("ambiguity_detection_noise", {})
    fr = adn.get("flip_rates", [])
    datasets = ["SQuAD", "AmbigQA", "TriviaQA-rc"]
    metrics_list = ["baseline_acc", "clar_acc", "avg_turns", "CES"]
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    axs_flat = axs.flatten()
    for i, met in enumerate(metrics_list):
        ax = axs_flat[i]
        for ds in datasets:
            vals = adn.get(ds, {}).get("metrics", {}).get(met, [])
            ax.plot(fr, vals, "o-", label=ds)
        ax.set_title(met.replace('_',' ').title() + " vs Flip Rate")
        ax.set_xlabel("Flip Rate"); ax.set_ylabel(met.replace('_',' ').title())
        ax.legend(); clean_axes(ax)
    fig.tight_layout()
    fig.savefig("figures/noise_detection_ablation.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 5:", e)

# === Figure 6: Post-Clarification Retrieval Noise (AmbigQA) ===
try:
    pcn = postclar_data.get("post_clar_noise", {})
    amb = pcn.get("AmbigQA", {}).get("metrics", {})
    noise = amb.get("noise_levels", [])
    b_acc = amb.get("baseline_acc", [])
    c_acc = amb.get("clar_acc", [])
    ces = amb.get("CES", [])
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].plot(noise, b_acc, "o-", label="Baseline")
    axs[0].plot(noise, c_acc, "s--", label="Clarified")
    axs[0].set_title("AmbigQA: Accuracy vs Retrieval Noise")
    axs[0].set_xlabel("Noise Level"); axs[0].set_ylabel("Accuracy")
    axs[0].legend(); clean_axes(axs[0])
    axs[1].plot(noise, ces, "o-", color="green")
    axs[1].set_title("AmbigQA: CES vs Retrieval Noise")
    axs[1].set_xlabel("Noise Level"); axs[1].set_ylabel("CES")
    clean_axes(axs[1])
    fig.tight_layout()
    fig.savefig("figures/post_retrieval_noise_AmbigQA.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 6:", e)

# === Figure 7: Confidence-Threshold Ablation (AmbigQA) ===
try:
    cta = confthr_data.get("confidence_threshold_ablation", {}).get("AmbigQA", {}).get("metrics", {})
    thr = cta.get("thresholds", [])
    base = cta.get("baseline_acc", [])
    clar = cta.get("clar_acc", [])
    turns = cta.get("avg_turns", [])
    ces = cta.get("CES", [])
    fig, axs = plt.subplots(3, 1, figsize=(6,12))
    # Accuracy
    axs[0].plot(thr, base, "o-", label="Baseline")
    axs[0].plot(thr, clar, "x--", label="Clarified")
    axs[0].set_title("AmbigQA: Accuracy vs Confidence Threshold")
    axs[0].set_xlabel("Threshold"); axs[0].set_ylabel("Accuracy")
    axs[0].legend(); clean_axes(axs[0])
    # Avg turns
    axs[1].plot(thr, turns, "s-", color="purple")
    axs[1].set_title("AmbigQA: Avg Turns vs Confidence Threshold")
    axs[1].set_xlabel("Threshold"); axs[1].set_ylabel("Avg Turns")
    clean_axes(axs[1])
    # CES
    axs[2].plot(thr, ces, "d-", color="red")
    axs[2].set_title("AmbigQA: CES vs Confidence Threshold")
    axs[2].set_xlabel("Threshold"); axs[2].set_ylabel("CES")
    clean_axes(axs[2])
    fig.tight_layout()
    fig.savefig("figures/confidence_threshold_AmbigQA.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 7:", e)

# === Figure 8: Question Format Ablation (AmbigQA) ===
try:
    formats = list(qform_data.keys())
    # Identify dataset-level metrics keys by inspecting first format
    ds_list = list(qform_data[formats[0]].keys()) if formats else []
    if "AmbigQA" in ds_list:
        baf, caf, atf, csf = [], [], [], []
        for f in formats:
            m = qform_data[f]["AmbigQA"]["metrics"]
            baf.append(m["baseline_acc"]); caf.append(m["clar_acc"])
            atf.append(m["avg_turns"]); csf.append(m["CES"])
        fig, axs = plt.subplots(3, 1, figsize=(6,12))
        x = np.arange(len(formats)); w = 0.35
        # Accuracy
        axs[0].bar(x - w/2, baf, w, label="Baseline")
        axs[0].bar(x + w/2, caf, w, label="Clar")
        axs[0].set_xticks(x); axs[0].set_xticklabels(formats, rotation=45)
        axs[0].set_title("AmbigQA: Accuracy by Question Format")
        axs[0].set_ylabel("Accuracy"); axs[0].legend()
        clean_axes(axs[0])
        # Avg turns
        axs[1].bar(formats, atf, color="gray")
        axs[1].set_title("AmbigQA: Avg Turns by Question Format")
        axs[1].set_ylabel("Avg Turns")
        clean_axes(axs[1])
        # CES
        axs[2].bar(formats, csf, color="orange")
        axs[2].set_title("AmbigQA: CES by Question Format")
        axs[2].set_ylabel("CES")
        clean_axes(axs[2])
        fig.tight_layout()
        fig.savefig("figures/question_format_AmbigQA.png")
        plt.close(fig)
except Exception as e:
    print("Error in Figure 8:", e)

# === Figure 9: Multi-Passage Answer Fusion (AmbigQA) ===
try:
    mpaf = mpaf_data.get("multi_passage_answer_fusion", {})
    mval = mpaf.get("AmbigQA", {}).get("metrics", {}).get("val", [])
    if len(mval) >= 2:
        wf, nf = mval[0], mval[1]
        fig, axs = plt.subplots(1,2, figsize=(12,5))
        # Accuracy comparison
        labels = ["Baseline", "Clar"]
        x = np.arange(len(labels)); w = 0.35
        axs[0].bar(x - w/2, [wf["baseline_acc"], wf["clar_acc"]], w, label="With Fusion")
        axs[0].bar(x + w/2, [nf["baseline_acc"], nf["clar_acc"]], w, label="No Fusion")
        axs[0].set_xticks(x); axs[0].set_xticklabels(labels)
        axs[0].set_title("AmbigQA: Accuracy With vs No Fusion")
        axs[0].set_ylabel("Accuracy"); axs[0].legend(); clean_axes(axs[0])
        # CES comparison
        axs[1].bar(["CES"], [wf["CES"]], w, label="With Fusion")
        axs[1].bar(["CES"], [nf["CES"]], w, label="No Fusion")
        axs[1].set_title("AmbigQA: CES With vs No Fusion")
        axs[1].set_ylabel("CES"); axs[1].legend(); clean_axes(axs[1])
        fig.tight_layout()
        fig.savefig("figures/multi_passage_fusion_AmbigQA.png")
        plt.close(fig)
except Exception as e:
    print("Error in Figure 9:", e)

# === Figure 10: User Patience Dropout (AmbigQA) ===
try:
    upd = patience_data.get("user_patience_dropout", {}).get("AmbigQA", {}).get("metrics", {})
    dr = upd.get("dropout_rates", [])
    ba = upd.get("baseline_acc", [])
    ca = upd.get("clar_acc", [])
    at = upd.get("avg_turns", [])
    ce = upd.get("CES", [])
    fig, axs = plt.subplots(3,1, figsize=(6,15))
    # Accuracy
    axs[0].plot(dr, ba, "o-", label="Baseline")
    axs[0].plot(dr, ca, "x--", label="Clar")
    axs[0].set_title("AmbigQA: Accuracy vs Dropout Rate")
    axs[0].set_xlabel("Dropout Rate"); axs[0].set_ylabel("Accuracy")
    axs[0].legend(); clean_axes(axs[0])
    # Avg turns
    axs[1].plot(dr, at, "s-", color="purple")
    axs[1].set_title("AmbigQA: Avg Turns vs Dropout Rate")
    axs[1].set_xlabel("Dropout Rate"); axs[1].set_ylabel("Avg Turns")
    clean_axes(axs[1])
    # CES
    axs[2].plot(dr, ce, "d-", color="red")
    axs[2].set_title("AmbigQA: CES vs Dropout Rate")
    axs[2].set_xlabel("Dropout Rate"); axs[2].set_ylabel("CES")
    clean_axes(axs[2])
    fig.tight_layout()
    fig.savefig("figures/user_patience_AmbigQA.png")
    plt.close(fig)
except Exception as e:
    print("Error in Figure 10:", e)

# === Figure 11 (Appendix): Always-Ask Clarification (AmbigQA) ===
try:
    aac = always_data.get("always_ask_clar", {}).get("AmbigQA", {}).get("metrics", {})
    ba = aac.get("baseline_acc", None)
    ca = aac.get("clar_acc", None)
    ce = aac.get("CES", None)
    if ba is not None:
        fig, axs = plt.subplots(1,2, figsize=(10,4))
        # Accuracy bar
        axs[0].bar(["Baseline","Clar"], [ba, ca], color=["skyblue","salmon"])
        axs[0].set_title("AmbigQA: Always-Ask Accuracy")
        axs[0].set_ylabel("Accuracy"); clean_axes(axs[0])
        # CES bar
        axs[1].bar(["CES"], [ce], color="orange")
        axs[1].set_title("AmbigQA: Always-Ask CES"); clean_axes(axs[1])
        fig.tight_layout()
        fig.savefig("figures/always_ask_AmbigQA.png")
        plt.close(fig)
except Exception as e:
    print("Error in Figure 11:", e)