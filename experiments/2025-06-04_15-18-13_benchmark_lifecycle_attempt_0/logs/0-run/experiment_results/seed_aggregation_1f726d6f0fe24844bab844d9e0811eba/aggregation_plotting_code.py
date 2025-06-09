import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load multiple experiment results
experiment_data_path_list = [
    "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_0cef0186366f4b18b3730b1f4d1d50db_proc_3702066/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_cdcac0528a7c4006b9bc7b3ac3c7e8b4_proc_3702065/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
    try:
        d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(d)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# Aggregate metrics across experiments
if all_experiment_data:
    train_mlp_list, train_cnn_list = [], []
    val_mlp_list, val_cnn_list = [], []
    orig_mlp_list, orig_cnn_list = [], []
    aug_mlp_list, aug_cnn_list = [], []
    cgr_list = []
    for data in all_experiment_data:
        base = data.get("original", {})
        lt = base.get("losses", {}).get("train", [])
        lv = base.get("losses", {}).get("val", [])
        oa = base.get("metrics", {}).get("orig_acc", [])
        aa = base.get("metrics", {}).get("aug_acc", [])
        cv = data.get("CGR", [])
        n_models = 2
        n_epochs = len(lt) // n_models if n_models else 0
        train_mlp_list.append(np.array(lt[0::2]))
        train_cnn_list.append(np.array(lt[1::2]))
        val_mlp_list.append(np.array(lv[0::2]))
        val_cnn_list.append(np.array(lv[1::2]))
        orig_mlp_list.append(np.array(oa[0::2]))
        orig_cnn_list.append(np.array(oa[1::2]))
        aug_mlp_list.append(np.array(aa[0::2]))
        aug_cnn_list.append(np.array(aa[1::2]))
        cgr_list.append(np.array(cv))

    # Stack and compute mean & SEM
    def mean_sem(arrs):
        a = np.vstack(arrs)
        m = a.mean(axis=0)
        s = a.std(axis=0, ddof=1) / np.sqrt(a.shape[0])
        return m, s

    train_mlp_mean, train_mlp_sem = mean_sem(train_mlp_list)
    train_cnn_mean, train_cnn_sem = mean_sem(train_cnn_list)
    val_mlp_mean, val_mlp_sem = mean_sem(val_mlp_list)
    val_cnn_mean, val_cnn_sem = mean_sem(val_cnn_list)
    orig_mlp_mean, orig_mlp_sem = mean_sem(orig_mlp_list)
    orig_cnn_mean, orig_cnn_sem = mean_sem(orig_cnn_list)
    aug_mlp_mean, aug_mlp_sem = mean_sem(aug_mlp_list)
    aug_cnn_mean, aug_cnn_sem = mean_sem(aug_cnn_list)
    cgr_mean, cgr_sem = mean_sem(cgr_list)
    epochs = np.arange(1, len(train_mlp_mean) + 1)
    # Print final aggregated metrics
    print("Final aggregated metrics:")
    print(f"MLP Train Loss: {train_mlp_mean[-1]:.4f} ± {train_mlp_sem[-1]:.4f}")
    print(f"CNN Train Loss: {train_cnn_mean[-1]:.4f} ± {train_cnn_sem[-1]:.4f}")
    print(f"MLP Orig Acc: {orig_mlp_mean[-1]:.4f} ± {orig_mlp_sem[-1]:.4f}")
    print(f"CNN Orig Acc: {orig_cnn_mean[-1]:.4f} ± {orig_cnn_sem[-1]:.4f}")
    print(f"Final CGR: {cgr_mean[-1]:.4f} ± {cgr_sem[-1]:.4f}")

    # Plot aggregated loss curves
    try:
        plt.figure()
        # MLP
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_mlp_mean, label="Train Loss Mean")
        plt.fill_between(
            epochs,
            train_mlp_mean - train_mlp_sem,
            train_mlp_mean + train_mlp_sem,
            alpha=0.2,
            label="Train Loss SEM",
        )
        plt.plot(epochs, val_mlp_mean, label="Val Loss Mean")
        plt.fill_between(
            epochs,
            val_mlp_mean - val_mlp_sem,
            val_mlp_mean + val_mlp_sem,
            alpha=0.2,
            label="Val Loss SEM",
        )
        plt.title("Left: MLP on MNIST")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # CNN
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_cnn_mean, label="Train Loss Mean")
        plt.fill_between(
            epochs,
            train_cnn_mean - train_cnn_sem,
            train_cnn_mean + train_cnn_sem,
            alpha=0.2,
            label="Train Loss SEM",
        )
        plt.plot(epochs, val_cnn_mean, label="Val Loss Mean")
        plt.fill_between(
            epochs,
            val_cnn_mean - val_cnn_sem,
            val_cnn_mean + val_cnn_sem,
            alpha=0.2,
            label="Val Loss SEM",
        )
        plt.title("Right: CNN on MNIST")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.suptitle("Aggregated Training and Validation Losses (MNIST)")
        plt.savefig(os.path.join(working_dir, "mnist_loss_curves_agg.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves plot: {e}")
        plt.close()

    # Plot aggregated accuracy curves
    try:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(epochs, orig_mlp_mean, label="Orig Acc Mean")
        plt.fill_between(
            epochs,
            orig_mlp_mean - orig_mlp_sem,
            orig_mlp_mean + orig_mlp_sem,
            alpha=0.2,
            label="Orig Acc SEM",
        )
        plt.plot(epochs, aug_mlp_mean, label="Aug Acc Mean")
        plt.fill_between(
            epochs,
            aug_mlp_mean - aug_mlp_sem,
            aug_mlp_mean + aug_mlp_sem,
            alpha=0.2,
            label="Aug Acc SEM",
        )
        plt.title("Left: MLP on MNIST")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, orig_cnn_mean, label="Orig Acc Mean")
        plt.fill_between(
            epochs,
            orig_cnn_mean - orig_cnn_sem,
            orig_cnn_mean + orig_cnn_sem,
            alpha=0.2,
            label="Orig Acc SEM",
        )
        plt.plot(epochs, aug_cnn_mean, label="Aug Acc Mean")
        plt.fill_between(
            epochs,
            aug_cnn_mean - aug_cnn_sem,
            aug_cnn_mean + aug_cnn_sem,
            alpha=0.2,
            label="Aug Acc SEM",
        )
        plt.title("Right: CNN on MNIST")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.suptitle("Aggregated Original and Augmented Accuracy (MNIST)")
        plt.savefig(os.path.join(working_dir, "mnist_accuracy_curves_agg.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy curves plot: {e}")
        plt.close()

    # Plot aggregated CGR curve
    try:
        plt.figure()
        epochs_cgr = np.arange(1, len(cgr_mean) + 1)
        plt.plot(epochs_cgr, cgr_mean, marker="o", label="CGR Mean")
        plt.fill_between(
            epochs_cgr,
            cgr_mean - cgr_sem,
            cgr_mean + cgr_sem,
            alpha=0.2,
            label="CGR SEM",
        )
        plt.title("CGR across Epochs on MNIST")
        plt.xlabel("Epoch")
        plt.ylabel("CGR")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "mnist_cgr_curve_agg.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CGR plot: {e}")
        plt.close()
