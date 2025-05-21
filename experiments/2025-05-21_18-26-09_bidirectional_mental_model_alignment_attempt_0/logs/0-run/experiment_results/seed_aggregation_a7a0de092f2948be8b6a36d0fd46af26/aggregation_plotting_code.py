import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load all experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_a01c5371854b472d9d698de9c013b3bb_proc_4081278/experiment_data.npy",
        "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_c54e9756eab849f398468a9dee659963_proc_4081280/experiment_data.npy",
        "experiments/2025-05-21_18-26-09_bidirectional_mental_model_alignment_attempt_0/logs/0-run/experiment_results/experiment_85f8d185ab7644bb8c36bce01d477867_proc_4081279/experiment_data.npy",
    ]
    all_experiment_data = []
    for rel_path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Prepare temperature keys and dataset list
temp_items = sorted(
    [(float(k.split("_")[-1]), k) for k in all_experiment_data[0].keys()]
)
first_key = temp_items[0][1]
datasets = list(all_experiment_data[0][first_key].keys())

# Plot aggregated per‐dataset metrics
for dataset in datasets:
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for idx, metric in enumerate(["losses", "accuracy", "alignments"]):
            ax = axs[idx]
            for temp, key in temp_items:
                # collect replicates
                y_tr_list = [
                    exp[key][dataset][metric]["train"] for exp in all_experiment_data
                ]
                y_val_list = [
                    exp[key][dataset][metric]["val"] for exp in all_experiment_data
                ]
                # truncate to shortest run
                min_epochs = min(len(y) for y in y_tr_list)
                y_tr_arr = np.array([y[:min_epochs] for y in y_tr_list])
                y_val_arr = np.array([y[:min_epochs] for y in y_val_list])
                # compute mean & SEM
                mean_tr = np.mean(y_tr_arr, axis=0)
                sem_tr = np.std(y_tr_arr, axis=0, ddof=1) / np.sqrt(len(y_tr_arr))
                mean_val = np.mean(y_val_arr, axis=0)
                sem_val = np.std(y_val_arr, axis=0, ddof=1) / np.sqrt(len(y_val_arr))
                epochs = np.arange(1, min_epochs + 1)
                # plot curves and SEM
                ax.plot(epochs, mean_tr, label=f"T={temp} train")
                ax.fill_between(epochs, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
                ax.plot(epochs, mean_val, linestyle="--", label=f"T={temp} val")
                ax.fill_between(
                    epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.2
                )
                # labels
                if metric == "losses":
                    ax.set_ylabel("Loss")
                elif metric == "accuracy":
                    ax.set_ylabel("Accuracy")
                else:
                    ax.set_ylabel("Alignment")
                ax.set_xlabel("Epoch")
                ax.set_title(f"{metric.capitalize()} Curves")
            ax.legend()
        fig.suptitle(f"{dataset} Aggregated Metrics across Temperatures")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{dataset}_aggregated_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated plot for {dataset}: {e}")
        plt.close()

# Summary plot of aggregated final MAI vs temperature
try:
    plt.figure()
    for dataset in datasets:
        temps = []
        mean_mais = []
        sem_mais = []
        for temp, key in temp_items:
            mai_vals = [exp[key][dataset]["mai"][-1] for exp in all_experiment_data]
            mean_mai = np.mean(mai_vals)
            sem_mai = np.std(mai_vals, ddof=1) / np.sqrt(len(mai_vals))
            temps.append(temp)
            mean_mais.append(mean_mai)
            sem_mais.append(sem_mai)
        plt.errorbar(
            temps, mean_mais, yerr=sem_mais, marker="o", linestyle="-", label=dataset
        )
    plt.xlabel("Softmax Temperature")
    plt.ylabel("Final MAI")
    plt.title("Aggregated Final Model Alignment-Accuracy Index vs Temperature")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_mai_vs_temperature.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated MAI summary plot: {e}")
    plt.close()

# Print out aggregated MAI summary
try:
    print("Aggregated final MAI per dataset (mean ± SEM):")
    for dataset in datasets:
        temps = []
        means = []
        sems = []
        for temp, key in temp_items:
            vals = [exp[key][dataset]["mai"][-1] for exp in all_experiment_data]
            temps.append(temp)
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        print(f"{dataset}: Temps {temps}")
        print(f"  Means: {np.round(means,4)}, SEM: {np.round(sems,4)}")
except Exception as e:
    print(f"Error printing MAI summary: {e}")
