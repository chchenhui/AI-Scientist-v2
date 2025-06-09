import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths to the three experiment data files
experiment_data_path_list = [
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_91521c88da8347558b30b105ff951d99_proc_103091/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_62e80975b1b143ad83eda1f52d796f6a_proc_103092/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_08ecc5ccc08e4289a6dd7b6b1701b055_proc_103093/experiment_data.npy",
]

# Load all experiments
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data from {path}: {e}")

# Identify all dataset names
dataset_names = set()
for ed in all_experiment_data:
    dataset_names.update(ed.keys())

# Aggregate and plot for each dataset
for ds in dataset_names:
    # Collect metrics across runs
    metrics_train_list = []
    metrics_val_list = []
    losses_train_list = []
    losses_val_list = []
    for ed in all_experiment_data:
        ds_data = ed.get(ds, {})
        metrics = ds_data.get("metrics", {})
        losses = ds_data.get("losses", {})
        if metrics.get("train") and metrics.get("val"):
            metrics_train_list.append(metrics["train"])
            metrics_val_list.append(metrics["val"])
        if losses.get("train") and losses.get("val"):
            losses_train_list.append(losses["train"])
            losses_val_list.append(losses["val"])

    # Plot aggregated metrics with mean ± SE
    if metrics_train_list:
        n_epochs = min(len(x) for x in metrics_train_list)
        train_arr = np.array([x[:n_epochs] for x in metrics_train_list])
        val_arr = np.array([x[:n_epochs] for x in metrics_val_list])
        epochs = np.arange(1, n_epochs + 1)
        mean_train = train_arr.mean(axis=0)
        se_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        mean_val = val_arr.mean(axis=0)
        se_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])

        # Print final epoch summary
        print(
            f"{ds} final metrics (Epoch {n_epochs}) - "
            f"Train Error Mean±SE: {mean_train[-1]:.4f}±{se_train[-1]:.4f}, "
            f"Val Error Mean±SE: {mean_val[-1]:.4f}±{se_val[-1]:.4f}"
        )

        try:
            plt.figure()
            plt.errorbar(
                epochs, mean_train, yerr=se_train, label="Train Error Mean ± SE"
            )
            plt.errorbar(epochs, mean_val, yerr=se_val, label="Val Error Mean ± SE")
            plt.xlabel("Epoch")
            plt.ylabel("Relative Error")
            plt.title(f"Training vs Validation Error (Mean ± SE)\nDataset: {ds}")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds}_metrics_mean_se.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating metrics plot for {ds}: {e}")
            plt.close()

    # Plot aggregated losses with mean ± SE
    if losses_train_list:
        n_epochs = min(len(x) for x in losses_train_list)
        train_loss_arr = np.array([x[:n_epochs] for x in losses_train_list])
        val_loss_arr = np.array([x[:n_epochs] for x in losses_val_list])
        epochs = np.arange(1, n_epochs + 1)
        mean_train_loss = train_loss_arr.mean(axis=0)
        se_train_loss = train_loss_arr.std(axis=0, ddof=1) / np.sqrt(
            train_loss_arr.shape[0]
        )
        mean_val_loss = val_loss_arr.mean(axis=0)
        se_val_loss = val_loss_arr.std(axis=0, ddof=1) / np.sqrt(val_loss_arr.shape[0])

        # Print final epoch summary
        print(
            f"{ds} final losses (Epoch {n_epochs}) - "
            f"Train Loss Mean±SE: {mean_train_loss[-1]:.4f}±{se_train_loss[-1]:.4f}, "
            f"Val Loss Mean±SE: {mean_val_loss[-1]:.4f}±{se_val_loss[-1]:.4f}"
        )

        try:
            plt.figure()
            plt.errorbar(
                epochs,
                mean_train_loss,
                yerr=se_train_loss,
                label="Train Loss Mean ± SE",
            )
            plt.errorbar(
                epochs, mean_val_loss, yerr=se_val_loss, label="Val Loss Mean ± SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title(f"Training vs Validation Loss (Mean ± SE)\nDataset: {ds}")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds}_losses_mean_se.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating losses plot for {ds}: {e}")
            plt.close()
