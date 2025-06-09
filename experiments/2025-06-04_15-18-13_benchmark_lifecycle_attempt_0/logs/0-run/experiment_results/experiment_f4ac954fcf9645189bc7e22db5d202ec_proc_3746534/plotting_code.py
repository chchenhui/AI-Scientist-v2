import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# iterate through noise settings and plot
for noise_key, noise_dict in data.get("label_noise", {}).items():
    try:
        eps_keys = sorted(noise_dict.keys(), key=lambda k: float(k.split("_")[1]))
        eps_vals = [float(k.split("_")[1]) for k in eps_keys]
        epochs = np.arange(1, len(noise_dict[eps_keys[0]]["losses"]["train"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for k, eps in zip(eps_keys, eps_vals):
            ed = noise_dict[k]
            tr = ed["losses"]["train"]
            vl = ed["losses"]["val"]
            oa = ed["metrics"]["orig_acc"]
            aa = ed["metrics"]["aug_acc"]
            axes[0].plot(epochs, tr, linestyle="solid", label=f"eps={eps}")
            axes[0].plot(epochs, vl, linestyle="dashed", label="_nolegend_")
            axes[1].plot(epochs, oa, linestyle="solid", label=f"eps={eps}")
            axes[1].plot(epochs, aa, linestyle="dashed", label="_nolegend_")
        axes[0].set_title("Training/Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(title="solid=Train, dashed=Val")
        axes[1].set_title("Original/Augmented Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(title="solid=Orig, dashed=Aug")
        fig.suptitle(f"MNIST (Noise={noise_key.split('_')[1]})")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"MNIST_noise_{noise_key.split('_')[1]}_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {noise_key}: {e}")
        plt.close()
