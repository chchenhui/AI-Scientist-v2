import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Prepare temperature keys and dataset list
temp_items = sorted([(float(k.split("_")[-1]), k) for k in experiment_data.keys()])
first_key = temp_items[0][1]
datasets = list(experiment_data[first_key].keys())

# Plot per‐dataset metrics
for dataset in datasets:
    try:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for idx, metric in enumerate(["losses", "accuracy", "alignments"]):
            ax = axs[idx]
            for temp, key in temp_items:
                rec = experiment_data[key][dataset]
                if metric == "losses":
                    y_tr, y_val = rec["losses"]["train"], rec["losses"]["val"]
                    ax.plot(range(1, len(y_tr) + 1), y_tr, label=f"T={temp} train")
                    ax.plot(
                        range(1, len(y_val) + 1),
                        y_val,
                        linestyle="--",
                        label=f"T={temp} val",
                    )
                    ax.set_ylabel("Loss")
                elif metric == "accuracy":
                    y_tr, y_val = rec["accuracy"]["train"], rec["accuracy"]["val"]
                    ax.plot(range(1, len(y_tr) + 1), y_tr, label=f"T={temp} train")
                    ax.plot(
                        range(1, len(y_val) + 1),
                        y_val,
                        linestyle="--",
                        label=f"T={temp} val",
                    )
                    ax.set_ylabel("Accuracy")
                else:  # alignments
                    y_tr, y_val = rec["alignments"]["train"], rec["alignments"]["val"]
                    ax.plot(range(1, len(y_tr) + 1), y_tr, label=f"T={temp} train")
                    ax.plot(
                        range(1, len(y_val) + 1),
                        y_val,
                        linestyle="--",
                        label=f"T={temp} val",
                    )
                    ax.set_ylabel("Alignment")
                ax.set_xlabel("Epoch")
                ax.set_title(f"{metric.capitalize()} Curves")
        fig.suptitle(f"{dataset} Metrics across Temperatures")
        fig.legend(loc="upper right")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{dataset}_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset}: {e}")
        plt.close()

# Summary plot of final MAI vs temperature
try:
    plt.figure()
    for dataset in datasets:
        temps, mai_vals = [], []
        for temp, key in temp_items:
            rec = experiment_data[key][dataset]
            temps.append(temp)
            mai_vals.append(rec["mai"][-1])
        plt.plot(temps, mai_vals, marker="o", label=dataset)
    plt.xlabel("Softmax Temperature")
    plt.ylabel("Final MAI")
    plt.title("Final Model Alignment-Accuracy Index vs Temperature")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mai_vs_temperature.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MAI summary plot: {e}")
    plt.close()
