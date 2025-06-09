import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

betas = [0.5, 0.7, 0.9, 0.99]
for dist, ed in experiment_data.get("noise_distribution", {}).items():
    # Plot error curves
    try:
        m = ed["metrics"]
        train_err = np.array(m["train"])
        val_err = np.array(m["val"])
        epochs = train_err.shape[1]
        x = np.arange(1, epochs + 1)
        plt.figure()
        for i, b in enumerate(betas):
            plt.plot(x, train_err[i], label=f"train β1={b}")
            plt.plot(x, val_err[i], "--", label=f"val β1={b}")
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Error")
        plt.title(f"Error Curves for {dist} (solid=train, dashed=val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dist}_error_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating error curves for {dist}: {e}")
        plt.close()
    # Plot loss curves
    try:
        l = ed["losses"]
        train_ls = np.array(l["train"])
        val_ls = np.array(l["val"])
        epochs = train_ls.shape[1]
        x = np.arange(1, epochs + 1)
        plt.figure()
        for i, b in enumerate(betas):
            plt.plot(x, train_ls[i], label=f"train β1={b}")
            plt.plot(x, val_ls[i], "--", label=f"val β1={b}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Loss Curves for {dist} (solid=train, dashed=val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dist}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dist}: {e}")
        plt.close()
