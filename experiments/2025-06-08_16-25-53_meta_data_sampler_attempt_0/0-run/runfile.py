import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_path_list = [
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_bd7b5ad309594c0db42f2ace277f4a76_proc_282857/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_23280bb328434303860d34b4941a7735_proc_282856/experiment_data.npy",
    "experiments/2025-06-08_16-25-53_meta_data_sampler_attempt_0/logs/0-run/experiment_results/experiment_81d13a3177094c1bbc55cc6fc98d42dd_proc_282858/experiment_data.npy",
]

all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        exp_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading experiment data from {experiment_data_path}: {e}")

if all_experiment_data:
    dataset_types = list(all_experiment_data[0].keys())
else:
    dataset_types = []

for ds_type in dataset_types:
    # Aggregated accuracy plot
    try:
        metrics_list = [
            exp[ds_type]["metrics"]
            for exp in all_experiment_data
            if ds_type in exp and "metrics" in exp[ds_type]
        ]
        if metrics_list:
            train_stack = np.array([m["train"] for m in metrics_list])
            val_stack = np.array([m["val"] for m in metrics_list])
            mean_train = train_stack.mean(axis=0)
            se_train = train_stack.std(axis=0, ddof=1) / np.sqrt(train_stack.shape[0])
            mean_val = val_stack.mean(axis=0)
            se_val = val_stack.std(axis=0, ddof=1) / np.sqrt(val_stack.shape[0])
            epochs = np.arange(len(mean_train))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"{ds_type} Aggregated Accuracy Curves")
            fig.text(
                0.5,
                0.92,
                "Left: Training Accuracy; Right: Validation Accuracy",
                ha="center",
            )
            ax1.errorbar(
                epochs, mean_train, yerr=se_train, capsize=3, label="Train Mean ±SE"
            )
            ax2.errorbar(epochs, mean_val, yerr=se_val, capsize=3, label="Val Mean ±SE")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.legend()
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()

            save_name = (
                f"{ds_type.lower().replace(' ', '_')}_aggregated_accuracy_curves.png"
            )
            plt.savefig(os.path.join(working_dir, save_name))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds_type}: {e}")
        plt.close()

    # Aggregated loss plot
    try:
        loss_list = [
            exp[ds_type]["losses"]
            for exp in all_experiment_data
            if ds_type in exp and "losses" in exp[ds_type]
        ]
        if loss_list:
            train_stack = np.array([l["train"] for l in loss_list])
            val_stack = np.array([l["val"] for l in loss_list])
            mean_train = train_stack.mean(axis=0)
            se_train = train_stack.std(axis=0, ddof=1) / np.sqrt(train_stack.shape[0])
            mean_val = val_stack.mean(axis=0)
            se_val = val_stack.std(axis=0, ddof=1) / np.sqrt(val_stack.shape[0])
            epochs = np.arange(len(mean_train))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"{ds_type} Aggregated Loss Curves")
            fig.text(
                0.5, 0.92, "Left: Training Loss; Right: Validation Loss", ha="center"
            )
            ax1.errorbar(
                epochs, mean_train, yerr=se_train, capsize=3, label="Train Mean ±SE"
            )
            ax2.errorbar(epochs, mean_val, yerr=se_val, capsize=3, label="Val Mean ±SE")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend()

            save_name = (
                f"{ds_type.lower().replace(' ', '_')}_aggregated_loss_curves.png"
            )
            plt.savefig(os.path.join(working_dir, save_name))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_type}: {e}")
        plt.close()
