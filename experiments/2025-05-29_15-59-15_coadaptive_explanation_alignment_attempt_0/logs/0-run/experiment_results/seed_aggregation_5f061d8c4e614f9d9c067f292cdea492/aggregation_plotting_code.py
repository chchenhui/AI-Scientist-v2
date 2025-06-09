import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Paths to experiment_data files relative to AI_SCIENTIST_ROOT
experiment_data_path_list = [
    "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_12bdbf98f1184a0393723ce3fe12841d_proc_2569236/experiment_data.npy",
    "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_7ad4f03fb5c54511b750217b607b4f1b_proc_2569238/experiment_data.npy",
    "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_bc59bb1520b04b6e831de7abd7f7da42_proc_2569237/experiment_data.npy",
]

# Load all experiment data
try:
    all_experiment_data = []
    for exp_path in experiment_data_path_list:
        loaded = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), exp_path), allow_pickle=True
        ).item()
        all_experiment_data.append(loaded)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Proceed only if data was loaded
if all_experiment_data:
    # Extract batch_size keys and select up to 5 evenly spaced
    keys = list(all_experiment_data[0]["batch_size"].keys())
    if len(keys) > 5:
        idxs = [int(i * (len(keys) - 1) / 4) for i in range(5)]
    else:
        idxs = list(range(len(keys)))
    selected_keys = [keys[i] for i in idxs]

    # Plot aggregated mean and SEM for each key
    for key in selected_keys:
        try:
            # Gather metrics and losses across runs
            train_accs = np.array(
                [
                    exp["batch_size"][key]["metrics"]["train"]
                    for exp in all_experiment_data
                ]
            )
            val_accs = np.array(
                [
                    exp["batch_size"][key]["metrics"]["val"]
                    for exp in all_experiment_data
                ]
            )
            train_losses = np.array(
                [
                    exp["batch_size"][key]["losses"]["train"]
                    for exp in all_experiment_data
                ]
            )
            val_losses = np.array(
                [exp["batch_size"][key]["losses"]["val"] for exp in all_experiment_data]
            )

            # Compute mean and standard error
            mean_train = train_accs.mean(axis=0)
            sem_train = train_accs.std(axis=0) / np.sqrt(train_accs.shape[0])
            mean_val = val_accs.mean(axis=0)
            sem_val = val_accs.std(axis=0) / np.sqrt(val_accs.shape[0])
            mean_l_train = train_losses.mean(axis=0)
            sem_l_train = train_losses.std(axis=0) / np.sqrt(train_losses.shape[0])
            mean_l_val = val_losses.mean(axis=0)
            sem_l_val = val_losses.std(axis=0) / np.sqrt(val_losses.shape[0])

            epochs = np.arange(1, mean_train.shape[0] + 1)
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            # Accuracy subplot
            axs[0].plot(epochs, mean_train, label="Train Mean", color="blue")
            axs[0].fill_between(
                epochs,
                mean_train - sem_train,
                mean_train + sem_train,
                color="blue",
                alpha=0.3,
                label="Train SEM",
            )
            axs[0].plot(epochs, mean_val, label="Val Mean", color="green")
            axs[0].fill_between(
                epochs,
                mean_val - sem_val,
                mean_val + sem_val,
                color="green",
                alpha=0.3,
                label="Val SEM",
            )
            axs[0].set_title("Accuracy")
            axs[0].set_xlabel("Epoch")
            axs[0].legend()

            # Loss subplot
            axs[1].plot(epochs, mean_l_train, label="Train Mean", color="red")
            axs[1].fill_between(
                epochs,
                mean_l_train - sem_l_train,
                mean_l_train + sem_l_train,
                color="red",
                alpha=0.3,
                label="Train SEM",
            )
            axs[1].plot(epochs, mean_l_val, label="Val Mean", color="orange")
            axs[1].fill_between(
                epochs,
                mean_l_val - sem_l_val,
                mean_l_val + sem_l_val,
                color="orange",
                alpha=0.3,
                label="Val SEM",
            )
            axs[1].set_title("Loss")
            axs[1].set_xlabel("Epoch")
            axs[1].legend()

            fig.suptitle(
                f"Synthetic binary dataset - batch size {key} (Left: Accuracy, Right: Loss)"
            )
            fname = os.path.join(
                working_dir, f"synthetic_binary_bs_{key}_aggregated_mean_sem.png"
            )
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated plot for {key}: {e}")
            plt.close()
