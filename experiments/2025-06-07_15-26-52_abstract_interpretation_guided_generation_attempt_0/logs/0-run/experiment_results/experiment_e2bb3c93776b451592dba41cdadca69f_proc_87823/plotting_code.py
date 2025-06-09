import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# Extract synthetic results for batch_size experiment
d = data.get("batch_size", {}).get("synthetic", {})
bs = d.get("params", [])
# Determine number of epochs
epochs = len(d.get("losses", {}).get("train", [[]])[0])

# Plot Loss Curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, b in enumerate(bs):
        axes[0].plot(range(1, epochs + 1), d["losses"]["train"][i], label=f"BS={b}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Synthetic Dataset - Training Loss")
    axes[0].legend()
    for i, b in enumerate(bs):
        axes[1].plot(range(1, epochs + 1), d["losses"]["val"][i], label=f"BS={b}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Synthetic Dataset - Validation Loss")
    axes[1].legend()
    fig.suptitle("Synthetic Dataset Loss Curves\nLeft: Training, Right: Validation")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot Pass Rates
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, b in enumerate(bs):
        axes[0].plot(range(1, epochs + 1), d["metrics"]["train"][i], label=f"BS={b}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Pass Rate")
    axes[0].set_title("Synthetic Dataset - Training Pass Rate")
    axes[0].legend()
    for i, b in enumerate(bs):
        axes[1].plot(range(1, epochs + 1), d["metrics"]["val"][i], label=f"BS={b}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Pass Rate")
    axes[1].set_title("Synthetic Dataset - Validation Pass Rate")
    axes[1].legend()
    fig.suptitle("Synthetic Dataset Pass Rates\nLeft: Training, Right: Validation")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "synthetic_pass_rates.png"))
    plt.close()
except Exception as e:
    print(f"Error creating pass-rate plot: {e}")
    plt.close()

# Plot Mean Iterations
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, b in enumerate(bs):
        axes[0].plot(range(1, epochs + 1), d["iterations"]["train"][i], label=f"BS={b}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Iterations")
    axes[0].set_title("Synthetic Dataset - Training Iterations")
    axes[0].legend()
    for i, b in enumerate(bs):
        axes[1].plot(range(1, epochs + 1), d["iterations"]["val"][i], label=f"BS={b}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Iterations")
    axes[1].set_title("Synthetic Dataset - Validation Iterations")
    axes[1].legend()
    fig.suptitle("Synthetic Dataset Mean Iterations\nLeft: Training, Right: Validation")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "synthetic_mean_iterations.png"))
    plt.close()
except Exception as e:
    print(f"Error creating iterations plot: {e}")
    plt.close()
