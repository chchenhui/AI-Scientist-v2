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
else:
    ed = experiment_data["batch_size"]["synthetic"]
    batch_sizes = ed["batch_sizes"]
    epochs = ed["epochs"]
    metrics = ed["metrics"]
    losses = ed["losses"]
    preds_list = ed["predictions"]
    gt = ed["ground_truth"]

    # Training Alignment vs Epochs
    try:
        fig, ax = plt.subplots()
        fig.suptitle("Synthetic dataset")
        for bs, vals in zip(batch_sizes, metrics["train"]):
            ax.plot(epochs, vals, label=f"bs={bs}")
        ax.set_title("Training Alignment vs Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alignment")
        ax.legend()
        fig.savefig(os.path.join(working_dir, "synthetic_alignment_train.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating alignment train plot: {e}")
        plt.close()

    # Validation Alignment vs Epochs
    try:
        fig, ax = plt.subplots()
        fig.suptitle("Synthetic dataset")
        for bs, vals in zip(batch_sizes, metrics["val"]):
            ax.plot(epochs, vals, label=f"bs={bs}")
        ax.set_title("Validation Alignment vs Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alignment")
        ax.legend()
        fig.savefig(os.path.join(working_dir, "synthetic_alignment_val.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating alignment val plot: {e}")
        plt.close()

    # Training Loss vs Epochs
    try:
        fig, ax = plt.subplots()
        fig.suptitle("Synthetic dataset")
        for bs, vals in zip(batch_sizes, losses["train"]):
            ax.plot(epochs, vals, label=f"bs={bs}")
        ax.set_title("Training Loss vs Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.savefig(os.path.join(working_dir, "synthetic_loss_train.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating loss train plot: {e}")
        plt.close()

    # Validation Loss vs Epochs
    try:
        fig, ax = plt.subplots()
        fig.suptitle("Synthetic dataset")
        for bs, vals in zip(batch_sizes, losses["val"]):
            ax.plot(epochs, vals, label=f"bs={bs}")
        ax.set_title("Validation Loss vs Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.savefig(os.path.join(working_dir, "synthetic_loss_val.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating loss val plot: {e}")
        plt.close()

    # Final Accuracy vs Batch Size
    acc_list = [np.mean(preds == gt) for preds in preds_list]
    print(f"Final accuracies per batch size: {dict(zip(batch_sizes, acc_list))}")
    try:
        fig, ax = plt.subplots()
        fig.suptitle("Synthetic dataset")
        ax.bar([str(bs) for bs in batch_sizes], acc_list)
        ax.set_title("Final Accuracy vs Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Accuracy")
        fig.savefig(os.path.join(working_dir, "synthetic_accuracy_batch_size.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating accuracy bar plot: {e}")
        plt.close()
