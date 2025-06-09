import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
else:
    ed = data["adam_beta1"]["synthetic"]
    train_errs = ed["metrics"]["train"]
    val_errs = ed["metrics"]["val"]
    train_losses = ed["losses"]["train"]
    val_losses = ed["losses"]["val"]
    beta1_list = [0.5, 0.7, 0.9, 0.99]
    epochs = len(train_errs[0])
    xs = np.arange(1, epochs + 1)

    try:
        plt.figure()
        for errs, b1 in zip(train_errs, beta1_list):
            plt.plot(xs, errs, label=f"β1={b1}")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.title("Training Error vs Epoch - synthetic dataset")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_training_error.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    try:
        plt.figure()
        for errs, b1 in zip(val_errs, beta1_list):
            plt.plot(xs, errs, label=f"β1={b1}")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.title("Validation Error vs Epoch - synthetic dataset")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_validation_error.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    try:
        plt.figure()
        for ls, b1 in zip(train_losses, beta1_list):
            plt.plot(xs, ls, label=f"β1={b1}")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.title("Training Loss vs Epoch - synthetic dataset")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    try:
        plt.figure()
        for ls, b1 in zip(val_losses, beta1_list):
            plt.plot(xs, ls, label=f"β1={b1}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE on Test")
        plt.title("Validation Loss vs Epoch - synthetic dataset")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_validation_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    try:
        final_vals = [v[-1] for v in val_errs]
        best_idx = int(np.argmin(final_vals))
        gt = ed["ground_truth"][best_idx][0]
        pr = ed["predictions"][best_idx][0]
        b1 = beta1_list[best_idx]
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(gt)
        ax1.set_title("Ground Truth Sample")
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(pr)
        ax2.set_title(f"Generated Sample (β1={b1})")
        plt.suptitle(
            "Sample Reconstruction Comparison - Left: Ground Truth, Right: Generated Samples - synthetic dataset"
        )
        plt.savefig(
            os.path.join(working_dir, f"synthetic_sample_reconstruction_beta1_{b1}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()
