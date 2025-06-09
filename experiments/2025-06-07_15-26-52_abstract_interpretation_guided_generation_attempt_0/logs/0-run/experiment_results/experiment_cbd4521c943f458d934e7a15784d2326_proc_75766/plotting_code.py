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

# Extract embedding dimensions
dims = []
if "embedding_dim_tuning" in experiment_data:
    dims = sorted(int(d) for d in experiment_data["embedding_dim_tuning"].keys())

# Determine epoch range
if dims:
    num_epochs = len(
        experiment_data["embedding_dim_tuning"][str(dims[0])]["losses"]["train"]
    )
    epochs = list(range(1, num_epochs + 1))
else:
    epochs = []

# Plot loss curves
try:
    plt.figure()
    for d in dims:
        logs = experiment_data["embedding_dim_tuning"][str(d)]
        L = logs["losses"]
        plt.plot(epochs, L["train"], label=f"Train Loss dim{d}")
        plt.plot(epochs, L["val"], "--", label=f"Val Loss dim{d}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (Synthetic Spec Dataset)\nTrain vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spec_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot error-free rate curves
try:
    plt.figure()
    for d in dims:
        logs = experiment_data["embedding_dim_tuning"][str(d)]
        M = logs["metrics"]
        plt.plot(epochs, M["train"], label=f"Train ER dim{d}")
        plt.plot(epochs, M["val"], "--", label=f"Val ER dim{d}")
    plt.xlabel("Epoch")
    plt.ylabel("Error-Free Rate")
    plt.title(
        "Error-Free Rate Curves (Synthetic Spec Dataset)\nTrain vs Validation Error-Free Rate"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spec_error_free_rate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error-free rate plot: {e}")
    plt.close()

# Print final evaluation metrics
for d in dims:
    logs = experiment_data["embedding_dim_tuning"][str(d)]
    ft_loss = logs["losses"]["train"][-1]
    fv_loss = logs["losses"]["val"][-1]
    ft_er = logs["metrics"]["train"][-1]
    fv_er = logs["metrics"]["val"][-1]
    print(
        f"Emb dim {d}: Final Train Loss={ft_loss:.4f}, Val Loss={fv_loss:.4f}, "
        f"Train ER={ft_er:.4f}, Val ER={fv_er:.4f}"
    )
