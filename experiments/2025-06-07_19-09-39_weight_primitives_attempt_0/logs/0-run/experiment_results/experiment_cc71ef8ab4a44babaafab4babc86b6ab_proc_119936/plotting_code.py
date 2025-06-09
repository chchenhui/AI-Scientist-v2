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
    experiment_data = None

if experiment_data:
    data = experiment_data["initialization"]["synthetic"]
    train_errs = np.array(data["metrics"]["train"])
    val_errs = np.array(data["metrics"]["val"])
    train_losses = np.array(data["losses"]["train"])
    val_losses = np.array(data["losses"]["val"])
    schemes = data["init_schemes"]
    init_D_list = sorted({s["D"] for s in schemes})

    for D in init_D_list:
        idxs = [i for i, s in enumerate(schemes) if s["D"] == D]
        print(
            f"Final Avg Validation Error for D={D}: " f"{val_errs[idxs, -1].mean():.4f}"
        )

    try:
        plt.figure()
        for D in init_D_list:
            idxs = [i for i, s in enumerate(schemes) if s["D"] == D]
            plt.plot(train_errs[idxs].mean(axis=0), label=f"D: {D}")
        plt.title("Synthetic Dataset - Training Error Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_training_error_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training error plot: {e}")
        plt.close()

    try:
        plt.figure()
        for D in init_D_list:
            idxs = [i for i, s in enumerate(schemes) if s["D"] == D]
            plt.plot(val_errs[idxs].mean(axis=0), label=f"D: {D}")
        plt.title("Synthetic Dataset - Validation Error Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_validation_error_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation error plot: {e}")
        plt.close()

    try:
        plt.figure()
        for D in init_D_list:
            idxs = [i for i, s in enumerate(schemes) if s["D"] == D]
            plt.plot(
                train_losses[idxs].mean(axis=0), linestyle="--", label=f"{D} Train"
            )
            plt.plot(val_losses[idxs].mean(axis=0), linestyle="-", label=f"{D} Val")
        plt.title("Synthetic Dataset - Training & Validation Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()
