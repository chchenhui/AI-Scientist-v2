import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths to all experiment_data.npy files
experiment_data_path_list = [
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/experiment_results/experiment_ff10ac06c81242499409da36b655cf2f_proc_3806/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/experiment_results/experiment_563867db33f741d7b58a23453bddaeb9_proc_3804/experiment_data.npy",
]

# Load all experiments
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading experiment data from {path}: {e}")

if not all_experiment_data:
    print("No experiment data loaded, exiting.")
else:
    # Identify datasets under "learning_rate"
    lr_group = all_experiment_data[0].get("learning_rate", {})
    for dataset_name in lr_group:
        # Stack metrics and losses across experiments
        try:
            lrs = np.array(lr_group[dataset_name]["lrs"])
            train_metrics = np.stack(
                [
                    exp["learning_rate"][dataset_name]["metrics"]["train"]
                    for exp in all_experiment_data
                ],
                axis=0,
            )
            val_metrics = np.stack(
                [
                    exp["learning_rate"][dataset_name]["metrics"]["val"]
                    for exp in all_experiment_data
                ],
                axis=0,
            )
            train_losses = np.stack(
                [
                    exp["learning_rate"][dataset_name]["losses"]["train"]
                    for exp in all_experiment_data
                ],
                axis=0,
            )
            val_losses = np.stack(
                [
                    exp["learning_rate"][dataset_name]["losses"]["val"]
                    for exp in all_experiment_data
                ],
                axis=0,
            )
            n_expts = train_metrics.shape[0]
            epochs_acc = np.arange(1, train_metrics.shape[2] + 1)
            epochs_loss = np.arange(1, train_losses.shape[2] + 1)
            mean_train_m = train_metrics.mean(axis=0)
            se_train_m = train_metrics.std(axis=0) / np.sqrt(n_expts)
            mean_val_m = val_metrics.mean(axis=0)
            se_val_m = val_metrics.std(axis=0) / np.sqrt(n_expts)
            mean_train_l = train_losses.mean(axis=0)
            se_train_l = train_losses.std(axis=0) / np.sqrt(n_expts)
            mean_val_l = val_losses.mean(axis=0)
            se_val_l = val_losses.std(axis=0) / np.sqrt(n_expts)
        except Exception as e:
            print(f"Error stacking data for dataset {dataset_name}: {e}")
            continue

        # Plot worst‐group accuracy with mean ± SE
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for i, lr in enumerate(lrs):
                axes[0].plot(epochs_acc, mean_train_m[i], label=f"lr={lr}")
                axes[0].fill_between(
                    epochs_acc,
                    mean_train_m[i] - se_train_m[i],
                    mean_train_m[i] + se_train_m[i],
                    alpha=0.2,
                )
                axes[1].plot(epochs_acc, mean_val_m[i], label=f"lr={lr}")
                axes[1].fill_between(
                    epochs_acc,
                    mean_val_m[i] - se_val_m[i],
                    mean_val_m[i] + se_val_m[i],
                    alpha=0.2,
                )
            axes[0].set_title("Training WG Accuracy")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Worst‐group Accuracy")
            axes[1].set_title("Validation WG Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Worst‐group Accuracy")
            for ax in axes:
                ax.legend()
            fig.suptitle(
                f"{dataset_name} dataset - Worst‐group Accuracy\nLeft: Training, Right: Validation"
            )
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_wg_accuracy_mean_se.png")
            )
            plt.close(fig)
        except Exception as e:
            print(f"Error creating WG accuracy plot for {dataset_name}: {e}")
            plt.close("all")

        # Plot loss curves with mean ± SE
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for i, lr in enumerate(lrs):
                axes[0].plot(epochs_loss, mean_train_l[i], label=f"lr={lr}")
                axes[0].fill_between(
                    epochs_loss,
                    mean_train_l[i] - se_train_l[i],
                    mean_train_l[i] + se_train_l[i],
                    alpha=0.2,
                )
                axes[1].plot(epochs_loss, mean_val_l[i], label=f"lr={lr}")
                axes[1].fill_between(
                    epochs_loss,
                    mean_val_l[i] - se_val_l[i],
                    mean_val_l[i] + se_val_l[i],
                    alpha=0.2,
                )
            axes[0].set_title("Training Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[1].set_title("Validation Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            for ax in axes:
                ax.legend()
            fig.suptitle(
                f"{dataset_name} dataset - Loss Curves\nLeft: Training, Right: Validation"
            )
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_loss_curves_mean_se.png")
            )
            plt.close(fig)
        except Exception as e:
            print(f"Error creating loss curves plot for {dataset_name}: {e}")
            plt.close("all")

        # Print final validation worst‐group accuracy means and SE
        final_means = mean_val_m[:, -1]
        final_ses = se_val_m[:, -1]
        print(f"Final Validation WG Accuracy for dataset '{dataset_name}':")
        for lr, m, s in zip(lrs, final_means, final_ses):
            print(f"  lr={lr}: {m:.3f} ± {s:.3f}")
