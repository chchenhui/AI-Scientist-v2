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

# Print final validation losses
for ablation in ["baseline", "continuous_memory"]:
    for ds_key, ds_data in experiment_data.get(ablation, {}).items():
        try:
            final_loss = ds_data["losses"]["val"][-1]
            print(f"{ablation} {ds_key} final val loss: {final_loss:.4f}")
        except Exception:
            pass

# Plotting metrics
for metric in [
    "losses",
    "Memory Retention Ratio",
    "Entropy-Weighted Memory Efficiency",
]:
    metric_fname = metric.lower().replace(" ", "_")
    for ds_key in experiment_data.get("baseline", {}):
        try:
            plt.figure()
            # Determine data paths
            if metric == "losses":
                b_train = experiment_data["baseline"][ds_key]["losses"]["train"]
                b_val = experiment_data["baseline"][ds_key]["losses"]["val"]
                c_train = experiment_data["continuous_memory"][ds_key]["losses"][
                    "train"
                ]
                c_val = experiment_data["continuous_memory"][ds_key]["losses"]["val"]
            else:
                b_train = experiment_data["baseline"][ds_key]["metrics"][metric][
                    "train"
                ]
                b_val = experiment_data["baseline"][ds_key]["metrics"][metric]["val"]
                c_train = experiment_data["continuous_memory"][ds_key]["metrics"][
                    metric
                ]["train"]
                c_val = experiment_data["continuous_memory"][ds_key]["metrics"][metric][
                    "val"
                ]
            epochs = range(1, len(b_train) + 1)
            # Plot lines
            plt.plot(epochs, b_train, label="Baseline Train")
            plt.plot(epochs, b_val, label="Baseline Val")
            plt.plot(epochs, c_train, label="ContMem Train")
            plt.plot(epochs, c_val, label="ContMem Val")
            # Labels and title
            plt.title(f"{metric} for {ds_key}")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            # Save figure
            fname = f"{metric_fname}_{ds_key}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating {metric} plot for {ds_key}: {e}")
            plt.close()
