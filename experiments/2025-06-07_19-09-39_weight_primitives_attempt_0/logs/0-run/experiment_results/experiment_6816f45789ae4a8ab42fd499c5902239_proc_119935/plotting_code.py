import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Extract synthetic results
ed = experiment_data.get("dictionary_capacity", {}).get("synthetic", {})
n_components_list = ed.get("n_components_list", [])
beta1_list = ed.get("beta1_list", [])
metrics_train = ed.get("metrics", {}).get("train", [])
metrics_val = ed.get("metrics", {}).get("val", [])
loss_train = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])

# Plot reconstruction error curves for each dictionary size
for idx, n_comp in enumerate(n_components_list):
    start = idx * len(beta1_list)
    try:
        plt.figure()
        for j, b1 in enumerate(beta1_list):
            plt.plot(metrics_train[start + j], label=f"Train β1={b1}")
            plt.plot(metrics_val[start + j], "--", label=f"Val β1={b1}")
        plt.title(f"Synthetic: Reconstruction Error Curves (n_components={n_comp})")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.legend()
        fname = f"synthetic_error_curves_nc{n_comp}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating error curves for n_components={n_comp}: {e}")
        plt.close()

# Plot loss curves for smallest and largest dictionary sizes
for idx in [0, len(n_components_list) - 1] if n_components_list else []:
    n_comp = n_components_list[idx]
    start = idx * len(beta1_list)
    try:
        plt.figure()
        for j, b1 in enumerate(beta1_list):
            plt.plot(loss_train[start + j], label=f"Train β1={b1}")
            plt.plot(loss_val[start + j], "--", label=f"Val β1={b1}")
        plt.title(f"Synthetic: MSE Loss Curves (n_components={n_comp})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        fname = f"synthetic_loss_curves_nc{n_comp}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for n_components={n_comp}: {e}")
        plt.close()
