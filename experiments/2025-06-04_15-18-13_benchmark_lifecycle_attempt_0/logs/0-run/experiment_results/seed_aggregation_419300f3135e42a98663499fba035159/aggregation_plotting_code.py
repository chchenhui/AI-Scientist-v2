import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data from multiple runs
try:
    experiment_data_path_list = [
        "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_d32afe54fdf04f4cbd80d2dfd0dd8a6c_proc_3732492/experiment_data.npy",
        "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_90838f2f141545c19d2fb014a7b330f7_proc_3732491/experiment_data.npy",
        "experiments/2025-06-04_15-18-13_benchmark_lifecycle_attempt_0/logs/0-run/experiment_results/experiment_48791ddcb8ba45efadffa28f5ca9991d_proc_3732490/experiment_data.npy",
    ]
    all_experiment_data = []
    for path in experiment_data_path_list:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Print aggregated final validation accuracies (mean ± SEM)
try:
    labels = list(all_experiment_data[0].keys())
    models = list(all_experiment_data[0][labels[0]]["metrics"].keys())
    for ds_name in labels:
        for model_name in models:
            vals = [
                run_data[ds_name]["metrics"][model_name]["val_acc"][-1]
                for run_data in all_experiment_data
            ]
            mean_val = np.mean(vals)
            sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals))
            print(
                f"{ds_name} - {model_name}: final validation accuracy = {mean_val:.4f} ± {sem_val:.4f}"
            )
except Exception as e:
    print(f"Error printing aggregated accuracy: {e}")

# Plot metrics curves with mean ± SEM per dataset
for ds_name in labels:
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        models = list(all_experiment_data[0][ds_name]["metrics"].keys())
        for model_name in models:
            # Determine minimum epoch length across runs
            run_lengths = [
                len(run_data[ds_name]["metrics"][model_name]["train_loss"])
                for run_data in all_experiment_data
            ]
            n_epochs = min(run_lengths)
            epochs = np.arange(1, n_epochs + 1)
            # Stack metrics
            train_losses = np.array(
                [
                    run_data[ds_name]["metrics"][model_name]["train_loss"][:n_epochs]
                    for run_data in all_experiment_data
                ]
            )
            val_losses = np.array(
                [
                    run_data[ds_name]["metrics"][model_name]["val_loss"][:n_epochs]
                    for run_data in all_experiment_data
                ]
            )
            val_accs = np.array(
                [
                    run_data[ds_name]["metrics"][model_name]["val_acc"][:n_epochs]
                    for run_data in all_experiment_data
                ]
            )
            # Compute mean and SEM
            mean_train = train_losses.mean(axis=0)
            sem_train = train_losses.std(axis=0, ddof=1) / np.sqrt(
                train_losses.shape[0]
            )
            mean_val = val_losses.mean(axis=0)
            sem_val = val_losses.std(axis=0, ddof=1) / np.sqrt(val_losses.shape[0])
            mean_acc = val_accs.mean(axis=0)
            sem_acc = val_accs.std(axis=0, ddof=1) / np.sqrt(val_accs.shape[0])
            # Plot loss curves
            ax1.plot(epochs, mean_train, label=f"{model_name} train mean")
            ax1.fill_between(
                epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.2
            )
            ax1.plot(epochs, mean_val, linestyle="--", label=f"{model_name} val mean")
            ax1.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.2)
            # Plot accuracy curves
            ax2.plot(epochs, mean_acc, label=f"{model_name} acc mean")
            ax2.fill_between(epochs, mean_acc - sem_acc, mean_acc + sem_acc, alpha=0.2)
        fig.suptitle(
            f"{ds_name.capitalize()} Metrics (Left: Loss Curves, Right: Accuracy Curves) Across Models - Mean ± SEM"
        )
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_title("Accuracy Curves")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax1.legend()
        ax2.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_metrics_mean_sem.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot for {ds_name}: {e}")
        plt.close()

# Plot discrimination score across datasets with mean ± SEM
try:
    plt.figure()
    for ds_name in labels:
        # Determine minimum length across runs
        lengths = [
            len(run_data[ds_name]["discrimination_score"])
            for run_data in all_experiment_data
        ]
        n_epochs = min(lengths)
        epochs = np.arange(1, n_epochs + 1)
        scores = np.array(
            [
                run_data[ds_name]["discrimination_score"][:n_epochs]
                for run_data in all_experiment_data
            ]
        )
        mean_score = scores.mean(axis=0)
        sem_score = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
        plt.plot(epochs, mean_score, label=ds_name)
        plt.fill_between(
            epochs, mean_score - sem_score, mean_score + sem_score, alpha=0.2
        )
    plt.title("Discrimination Score Across Datasets - Mean ± SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Discrimination Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "discrimination_score_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating discrimination score plot: {e}")
    plt.close()

# Plot final validation accuracy comparison with SEM
try:
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    for i, model_name in enumerate(models):
        means = []
        sems = []
        for ds_name in labels:
            vals = [
                run_data[ds_name]["metrics"][model_name]["val_acc"][-1]
                for run_data in all_experiment_data
            ]
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.bar(x + i * width, means, width, yerr=sems, capsize=5, label=model_name)
    ax.set_title("Final Validation Accuracy Comparison Across Datasets - Mean ± SEM")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(os.path.join(working_dir, "final_val_accuracy_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy comparison plot: {e}")
    plt.close()
