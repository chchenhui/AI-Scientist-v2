import os
import numpy as np
import matplotlib.pyplot as plt

# Configure font size for publication
plt.rcParams.update({'font.size': 12})

# Make output dirs
os.makedirs("figures", exist_ok=True)
os.makedirs("figures/appendix", exist_ok=True)

# 1) Baseline: Training/validation error and loss, plus best-sample comparison
try:
    npy = ("experiment_results/"
           "experiment_6effbbcb54b241c9b3db94d9c6486930_proc_106393/"
           "experiment_data.npy")
    bd = np.load(npy, allow_pickle=True).item()['adam_beta1']['synthetic']
    betas = [0.5, 0.7, 0.9, 0.99]
    tr_err = bd['metrics']['train']
    val_err = bd['metrics']['val']
    tr_loss = bd['losses']['train']
    val_loss = bd['losses']['val']
    epochs = len(tr_err[0])
    x = np.arange(1, epochs+1)

    # Figure 1: 2x2 grid – train/val error & loss
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    # Training error
    for errs, b in zip(tr_err, betas):
        axs[0,0].plot(x, errs, label=f"β₁={b}")
    axs[0,0].set_title("Training Relative Error")
    axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("Relative Error")
    axs[0,0].legend()
    # Validation error
    for errs, b in zip(val_err, betas):
        axs[0,1].plot(x, errs, label=f"β₁={b}")
    axs[0,1].set_title("Validation Relative Error")
    axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Relative Error")
    axs[0,1].legend()
    # Training loss
    for ls, b in zip(tr_loss, betas):
        axs[1,0].plot(x, ls, label=f"β₁={b}")
    axs[1,0].set_title("Training Reconstruction Loss")
    axs[1,0].set_xlabel("Epoch"); axs[1,0].set_ylabel("MSE Loss")
    axs[1,0].legend()
    # Validation loss
    for ls, b in zip(val_loss, betas):
        axs[1,1].plot(x, ls, label=f"β₁={b}")
    axs[1,1].set_title("Validation Reconstruction Loss")
    axs[1,1].set_xlabel("Epoch"); axs[1,1].set_ylabel("MSE Loss")
    axs[1,1].legend()
    plt.tight_layout()
    fig.savefig("figures/baseline_curves.png")
    plt.close(fig)

    # Figure 2: Best-sample reconstruction
    final_val = [v[-1] for v in val_err]
    best = int(np.argmin(final_val))
    gt = bd['ground_truth'][best][0]
    pr = bd['predictions'][best][0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    ax[0].plot(gt, color='black')
    ax[0].set_title("Ground Truth Sample")
    ax[0].set_xlabel("Dimension")
    ax[1].plot(pr, color='steelblue')
    ax[1].set_title(f"Reconstructed Sample (β₁={betas[best]})")
    ax[1].set_xlabel("Dimension")
    plt.suptitle("Baseline Sample Reconstruction")
    plt.tight_layout(rect=[0,0,1,0.92])
    fig.savefig("figures/baseline_sample.png")
    plt.close(fig)

except Exception as e:
    print("Baseline plotting error:", e)


# 2) Multi-synthetic robustness (ds1, ds2, ds3)
try:
    npy = ("experiment_results/"
           "experiment_cb095268033542b0a36a5eb8d437fc1e_proc_119934/"
           "experiment_data.npy")
    ms = np.load(npy, allow_pickle=True).item()['multi_synthetic']
    names = list(ms.keys())
    rows = len(names)
    fig, axs = plt.subplots(rows, 2, figsize=(12, 4*rows), dpi=300)
    for i, name in enumerate(names):
        ed = ms[name]
        tr_e = ed['metrics']['train']
        val_e = ed['metrics']['val']
        tr_l = ed['losses']['train']
        val_l = ed['losses']['val']
        # Error plot
        for errs, b in zip(tr_e, betas):
            axs[i,0].plot(errs, label=f"train β₁={b}")
        for errs, b in zip(val_e, betas):
            axs[i,0].plot(errs, '--', label=f"val β₁={b}")
        axs[i,0].set_title(f"{name}: Error Curves")
        axs[i,0].set_xlabel("Epoch"); axs[i,0].set_ylabel("Relative Error")
        axs[i,0].legend(fontsize=9)
        # Loss plot
        for ls, b in zip(tr_l, betas):
            axs[i,1].plot(ls, label=f"train β₁={b}")
        for ls, b in zip(val_l, betas):
            axs[i,1].plot(ls, '--', label=f"val β₁={b}")
        axs[i,1].set_title(f"{name}: Loss Curves")
        axs[i,1].set_xlabel("Epoch"); axs[i,1].set_ylabel("MSE Loss")
        axs[i,1].legend(fontsize=9)
    plt.tight_layout()
    fig.suptitle("Multi-Synthetic Dataset Robustness", fontsize=14)
    plt.subplots_adjust(top=0.93)
    fig.savefig("figures/multi_synthetic.png")
    plt.close(fig)

except Exception as e:
    print("Multi-synthetic plotting error:", e)


# 3) Dictionary capacity ablation (n_components=10 and 60)
try:
    npy = ("experiment_results/"
           "experiment_6816f45789ae4a8ab42fd499c5902239_proc_119935/"
           "experiment_data.npy")
    dc = np.load(npy, allow_pickle=True).item()['dictionary_capacity']['synthetic']
    comps = dc['n_components_list']
    tr_e = dc['metrics']['train']
    val_e = dc['metrics']['val']
    tr_l = dc['losses']['train']
    val_l = dc['losses']['val']
    # Select first and last component settings
    sel = [0, len(comps)-1]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    for row, idx in enumerate(sel):
        c = comps[idx]
        base = idx * len(betas)
        # error
        for j, b in enumerate(betas):
            axs[row,0].plot(tr_e[base+j], label=f"train β₁={b}")
            axs[row,0].plot(val_e[base+j], '--', label=f"val β₁={b}")
        axs[row,0].set_title(f"n_components={c} Error")
        axs[row,0].set_xlabel("Epoch"); axs[row,0].set_ylabel("Relative Error")
        axs[row,0].legend(fontsize=8)
        # loss
        for j, b in enumerate(betas):
            axs[row,1].plot(tr_l[base+j], label=f"train β₁={b}")
            axs[row,1].plot(val_l[base+j], '--', label=f"val β₁={b}")
        axs[row,1].set_title(f"n_components={c} Loss")
        axs[row,1].set_xlabel("Epoch"); axs[row,1].set_ylabel("MSE Loss")
        axs[row,1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig("figures/dictionary_capacity.png")
    plt.close(fig)

except Exception as e:
    print("Dictionary capacity plotting error:", e)


# 4) Optimizer choice ablation (SGD, RMSprop, AdamW)
try:
    npy = ("experiment_results/"
           "experiment_2168551bfbbb4d118c9fce27ad0af6fb_proc_119934/"
           "experiment_data.npy")
    od = np.load(npy, allow_pickle=True).item()['optimizer_choice']['synthetic']
    opts = od['optimizers']
    tr_e = np.array(od['metrics']['train'])
    val_e = np.array(od['metrics']['val'])
    tr_l = np.array(od['losses']['train'])
    val_l = np.array(od['losses']['val'])

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    # training error
    for i,opt in enumerate(opts):
        axs[0,0].plot(tr_e[i], label=opt)
    axs[0,0].set_title("Training Error by Optimizer")
    axs[0,0].set_xlabel("Epoch"); axs[0,0].set_ylabel("Relative Error")
    axs[0,0].legend(fontsize=9)
    # validation error
    for i,opt in enumerate(opts):
        axs[0,1].plot(val_e[i], label=opt)
    axs[0,1].set_title("Validation Error by Optimizer")
    axs[0,1].set_xlabel("Epoch"); axs[0,1].set_ylabel("Relative Error")
    axs[0,1].legend(fontsize=9)
    # training loss
    for i,opt in enumerate(opts):
        axs[1,0].plot(tr_l[i], label=opt)
    axs[1,0].set_title("Training Loss by Optimizer")
    axs[1,0].set_xlabel("Epoch"); axs[1,0].set_ylabel("MSE Loss")
    axs[1,0].legend(fontsize=9)
    # validation loss
    for i,opt in enumerate(opts):
        axs[1,1].plot(val_l[i], label=opt)
    axs[1,1].set_title("Validation Loss by Optimizer")
    axs[1,1].set_xlabel("Epoch"); axs[1,1].set_ylabel("MSE Loss")
    axs[1,1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig("figures/optimizer_choice.png")
    plt.close(fig)

except Exception as e:
    print("Optimizer choice plotting error:", e)


# 5) Initialization scheme ablation
try:
    npy = ("experiment_results/"
           "experiment_cc71ef8ab4a44babaafab4babc86b6ab_proc_119936/"
           "experiment_data.npy")
    idata = np.load(npy, allow_pickle=True).item()['initialization']['synthetic']
    tr_e = np.array(idata['metrics']['train'])
    val_e = np.array(idata['metrics']['val'])
    tr_l = np.array(idata['losses']['train'])
    val_l = np.array(idata['losses']['val'])
    schemes = idata['init_schemes']
    Ds = sorted({s['D'] for s in schemes})

    # average across code inits per D
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    for D in Ds:
        idxs = [i for i,s in enumerate(schemes) if s['D']==D]
        mean_tr_e = tr_e[idxs].mean(axis=0)
        mean_val_e = val_e[idxs].mean(axis=0)
        mean_tr_l = tr_l[idxs].mean(axis=0)
        mean_val_l = val_l[idxs].mean(axis=0)
        label = f"D={D}"
        axs[0,0].plot(mean_tr_e, label=label)
        axs[0,1].plot(mean_val_e, label=label)
        axs[1,0].plot(mean_tr_l, label=label)
        axs[1,1].plot(mean_val_l, label=label)
    axs[0,0].set_title("Training Error by Init D"); axs[0,0].set_ylabel("Relative Error")
    axs[0,1].set_title("Validation Error by Init D"); axs[0,1].set_ylabel("Relative Error")
    axs[1,0].set_title("Training Loss by Init D"); axs[1,0].set_ylabel("MSE Loss")
    axs[1,1].set_title("Validation Loss by Init D"); axs[1,1].set_ylabel("MSE Loss")
    for ax in axs.flatten(): ax.set_xlabel("Epoch"); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("figures/initialization_scheme.png")
    plt.close(fig)

except Exception as e:
    print("Initialization scheme plotting error:", e)


# 6) Mini-batch size ablation
try:
    npy = ("experiment_results/"
           "experiment_df9171d7253c4881af69d69a29bdaaa5_proc_119934/"
           "experiment_data.npy")
    mb = np.load(npy, allow_pickle=True).item()['mini_batch_size']['synthetic']
    bss = mb['batch_sizes']
    tr_e = mb['metrics']['train']
    val_e = mb['metrics']['val']
    tr_l = mb['losses']['train']
    val_l = mb['losses']['val']

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    # train error
    for errs, bs in zip(tr_e, bss):
        axs[0,0].plot(errs, label=f"bs={bs}")
    axs[0,0].set_title("Training Error"); axs[0,0].set_ylabel("Relative Error")
    # val error
    for errs, bs in zip(val_e, bss):
        axs[0,1].plot(errs, label=f"bs={bs}")
    axs[0,1].set_title("Validation Error"); axs[0,1].set_ylabel("Relative Error")
    # train loss
    for ls, bs in zip(tr_l, bss):
        axs[1,0].plot(ls, label=f"bs={bs}")
    axs[1,0].set_title("Training Loss"); axs[1,0].set_ylabel("MSE Loss")
    # val loss
    for ls, bs in zip(val_l, bss):
        axs[1,1].plot(ls, label=f"bs={bs}")
    axs[1,1].set_title("Validation Loss"); axs[1,1].set_ylabel("MSE Loss")
    for ax in axs.flatten(): ax.set_xlabel("Epoch"); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("figures/mini_batch_size.png")
    plt.close(fig)

except Exception as e:
    print("Mini-batch size plotting error:", e)


# Appendix: Noise-level ablation (σ levels)
try:
    npy = ("experiment_results/"
           "experiment_d0e45c415b7945c39648245ce5a9efaf_proc_119936/"
           "experiment_data.npy")
    nd = np.load(npy, allow_pickle=True).item()['synthetic_noise']['synthetic']
    sigmas = nd['noise_levels']
    tr_e = np.array(nd['metrics']['train'])
    val_e = np.array(nd['metrics']['val'])
    tr_l = np.array(nd['losses']['train'])
    val_l = np.array(nd['losses']['val'])
    x = np.arange(1, tr_e.shape[1]+1)

    # Error curves
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)
    for i, σ in enumerate(sigmas):
        ax.plot(x, tr_e[i], label=f"train σ={σ}")
        ax.plot(x, val_e[i], '--', label=f"val σ={σ}")
    ax.set_title("Error vs Noise Level")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Relative Error")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("figures/appendix/noise_error.png")
    plt.close(fig)

    # Loss curves
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)
    for i, σ in enumerate(sigmas):
        ax.plot(x, tr_l[i], label=f"train σ={σ}")
        ax.plot(x, val_l[i], '--', label=f"val σ={σ}")
    ax.set_title("Loss vs Noise Level")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("figures/appendix/noise_loss.png")
    plt.close(fig)

except Exception as e:
    print("Noise-level plotting error:", e)