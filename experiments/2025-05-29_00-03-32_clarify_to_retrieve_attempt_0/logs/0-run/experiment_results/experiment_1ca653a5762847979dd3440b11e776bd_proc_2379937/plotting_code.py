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

# Extract synthetic XOR results
data = experiment_data.get("hidden_layer_size", {}).get("synthetic_xor", {})
sizes = data.get("sizes", [])
loss_tr = data.get("losses", {}).get("train", [])
loss_val = data.get("losses", {}).get("val", [])
ces_tr = data.get("metrics", {}).get("train", [])
ces_val = data.get("metrics", {}).get("val", [])

# Print summary of final CES
print("Hidden sizes:", sizes)
print("Final train CES per size:", [m[-1] if m else None for m in ces_tr])
print("Final val CES per size:", [m[-1] if m else None for m in ces_val])

# Plot loss curves
try:
    plt.figure()
    for sz, lt, lv in zip(sizes, loss_tr, loss_val):
        plt.plot(range(1, len(lt) + 1), lt, label=f"Train loss (size={sz})")
        plt.plot(range(1, len(lv) + 1), lv, "--", label=f"Val loss (size={sz})")
    plt.title("Synthetic XOR: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot CES curves
try:
    plt.figure()
    for sz, mt, mv in zip(sizes, ces_tr, ces_val):
        plt.plot(range(1, len(mt) + 1), mt, label=f"Train CES (size={sz})")
        plt.plot(range(1, len(mv) + 1), mv, "--", label=f"Val CES (size={sz})")
    plt.title("Synthetic XOR: Training vs Validation CES")
    plt.xlabel("Epoch")
    plt.ylabel("CES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_CES_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES plot: {e}")
    plt.close()

# Plot final CES bar chart
try:
    plt.figure()
    plt.bar([str(s) for s in sizes], [m[-1] if m else 0 for m in ces_val])
    plt.title("Synthetic XOR: Final Validation CES by Hidden Size")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("CES")
    plt.savefig(os.path.join(working_dir, "synthetic_xor_final_CES_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final CES bar plot: {e}")
    plt.close()
