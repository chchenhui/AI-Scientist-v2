import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["dropout_rate_tuning"]["synthetic_xor"]
    dropouts = data["dropout_rates"]
    loss_train = data["losses"]["train"]
    loss_val = data["losses"]["val"]
    ces_train = data["metrics"]["train"]
    ces_val = data["metrics"]["val"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot CES vs Dropout Rate
try:
    plt.figure()
    plt.plot(dropouts, [c[-1] for c in ces_train], marker="o", label="Train CES")
    plt.plot(dropouts, [c[-1] for c in ces_val], marker="o", label="Val CES")
    plt.xlabel("Dropout Rate")
    plt.ylabel("CES")
    plt.title("CES vs Dropout Rate (synthetic_xor)\nLeft: Train CES, Right: Val CES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_CES_vs_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES plot: {e}")
    plt.close()

# Plot final Loss vs Dropout Rate
try:
    plt.figure()
    plt.plot(dropouts, [l[-1] for l in loss_train], marker="o", label="Train Loss")
    plt.plot(dropouts, [l[-1] for l in loss_val], marker="o", label="Val Loss")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Loss")
    plt.title(
        "Final Loss vs Dropout Rate (synthetic_xor)\nLeft: Train Loss, Right: Val Loss"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_final_loss_vs_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss vs dropout plot: {e}")
    plt.close()

# Plot Loss Curves vs Epoch for up to 5 dropout rates
try:
    plt.figure()
    epochs = len(loss_train[0])
    idxs = np.linspace(0, len(dropouts) - 1, min(len(dropouts), 5), dtype=int)
    for i in idxs:
        dr = dropouts[i]
        plt.plot(range(1, epochs + 1), loss_train[i], label=f"Train dr={dr}")
        plt.plot(
            range(1, epochs + 1), loss_val[i], linestyle="--", label=f"Val dr={dr}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (synthetic_xor)\nSolid: Train, Dashed: Val")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Print summary metrics
try:
    print("Dropout Rates:", dropouts)
    print("Final Train CES:", [c[-1] for c in ces_train])
    print("Final Val CES:", [c[-1] for c in ces_val])
except Exception as e:
    print(f"Error printing summary metrics: {e}")
