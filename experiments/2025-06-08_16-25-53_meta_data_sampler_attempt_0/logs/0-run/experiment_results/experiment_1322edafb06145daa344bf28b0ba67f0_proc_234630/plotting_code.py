import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load data
try:
    ed = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# 1) Plot avg train/val loss curves
try:
    mdata = ed["weight_decay"]["synthetic"]["metrics"]
    train = np.array(mdata["train"])  # shape (runs, epochs)
    val = np.array(mdata["val"])
    epochs = np.arange(1, train.shape[1] + 1)
    mean_train = train.mean(axis=0)
    mean_val = val.mean(axis=0)
    plt.figure()
    plt.plot(epochs, mean_train, "-o")
    plt.plot(epochs, mean_val, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Synthetic Dataset - Training and Validation Loss Curves")
    plt.legend(["Train Loss", "Val Loss"])
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2) Heatmap of final validation loss over weight decay grid
try:
    wd_main = np.array(ed["weight_decay"]["synthetic"]["weight_decay_main"], float)
    wd_dvn = np.array(ed["weight_decay"]["synthetic"]["weight_decay_dvn"], float)
    final_val = np.array(ed["weight_decay"]["synthetic"]["metrics"]["val"])[:, -1]
    um = np.unique(wd_main)
    ud = np.unique(wd_dvn)
    mat = final_val.reshape(len(um), len(ud))
    plt.figure()
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(np.arange(len(ud)), [str(d) for d in ud])
    plt.yticks(np.arange(len(um)), [str(m) for m in um])
    plt.xlabel("Weight Decay DVN")
    plt.ylabel("Weight Decay Main")
    plt.title("Synthetic Dataset - Final Validation Loss Heatmap")
    plt.savefig(os.path.join(working_dir, "synthetic_val_loss_heatmap.png"))
    plt.close()
except Exception as e:
    print(f"Error creating heatmap: {e}")
    plt.close()

# 3) Spearman correlation vs epoch
try:
    preds = np.array(
        ed["weight_decay"]["synthetic"]["predictions"]
    )  # (runs, epochs, samples)
    gts = np.array(ed["weight_decay"]["synthetic"]["ground_truth"])
    runs, epochs_n, _ = preds.shape

    def spearman(a, b):
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        return np.corrcoef(ar, br)[0, 1]

    corr_mat = np.zeros((runs, epochs_n))
    for i in range(runs):
        for j in range(epochs_n):
            corr_mat[i, j] = spearman(preds[i, j], gts[i, j])
    mean_corr = corr_mat.mean(axis=0)
    plt.figure()
    plt.plot(np.arange(1, epochs_n + 1), mean_corr, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.title("Synthetic Dataset - DVN Spearman Corr vs Epoch")
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_corr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Spearman plot: {e}")
    plt.close()
