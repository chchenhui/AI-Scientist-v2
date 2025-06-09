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
    data = {}
exp = data.get("Ablate_Meta_Inner_Update_Steps", {})

for name, res in exp.items():
    metrics = res.get("metrics", {})
    losses = res.get("losses", {})
    corrs = res.get("corrs", [])
    n_meta = res.get("N_meta_history", [])
    train_acc = metrics.get("train", [])
    val_acc = metrics.get("val", [])
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])

    # Accuracy over epochs
    try:
        plt.figure()
        plt.plot(train_acc, marker="o", label="Train Acc")
        plt.plot(val_acc, marker="s", label="Val Acc")
        plt.title(f"{name} Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_accuracy_curve.png"))
    except Exception as e:
        print(f"Error creating accuracy plot for {name}: {e}")
    finally:
        plt.close()

    # Loss over epochs
    try:
        plt.figure()
        plt.plot(train_loss, marker="o", label="Train Loss")
        plt.plot(val_loss, marker="s", label="Val Loss")
        plt.title(f"{name} Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curve.png"))
    except Exception as e:
        print(f"Error creating loss plot for {name}: {e}")
    finally:
        plt.close()

    # Spearman correlation over meta updates
    try:
        plt.figure()
        plt.plot(corrs, marker="o")
        plt.title(f"{name} Spearman Corr over Meta Updates")
        plt.xlabel("Meta Update Event Index")
        plt.ylabel("Spearman Correlation")
        plt.savefig(os.path.join(working_dir, f"{name}_spearman_corr.png"))
    except Exception as e:
        print(f"Error creating correlation plot for {name}: {e}")
    finally:
        plt.close()

    # N_meta history over meta updates
    try:
        plt.figure()
        plt.plot(n_meta, marker="o")
        plt.title(f"{name} N_meta History over Meta Updates")
        plt.xlabel("Meta Update Event Index")
        plt.ylabel("N_meta")
        plt.savefig(os.path.join(working_dir, f"{name}_n_meta_history.png"))
    except Exception as e:
        print(f"Error creating N_meta history plot for {name}: {e}")
    finally:
        plt.close()

    if val_acc:
        print(f"{name} Final Val Acc: {val_acc[-1]:.4f}")
    if val_loss:
        print(f"{name} Final Val Loss: {val_loss[-1]:.4f}")
