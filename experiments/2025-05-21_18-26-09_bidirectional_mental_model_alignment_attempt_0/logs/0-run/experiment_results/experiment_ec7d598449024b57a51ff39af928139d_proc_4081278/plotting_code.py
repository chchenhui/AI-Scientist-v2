import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# get sorted dropout keys and dataset names
drop_keys = sorted(experiment_data.keys(), key=lambda x: int(x.split("_")[-1]))
dataset_names = list(experiment_data[drop_keys[0]].keys()) if drop_keys else []

# plot accuracy curves per dataset
for ds in dataset_names:
    try:
        plt.figure()
        for key in drop_keys:
            rate = int(key.split("_")[-1])
            acc_tr = experiment_data[key][ds]["metrics"]["train"]
            acc_val = experiment_data[key][ds]["metrics"]["val"]
            epochs = np.arange(1, len(acc_tr) + 1)
            plt.plot(epochs, acc_tr, label=f"Train {rate}% dropout")
            plt.plot(epochs, acc_val, "--", label=f"Val {rate}% dropout")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f'{ds.replace("_"," ").title()} Classification\nTrain (solid) vs Val (dashed) Accuracy'
        )
        plt.legend()
        fname = f"{ds}_accuracy_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds}: {e}")
        plt.close()

# plot MAI vs dropout rate for all datasets
try:
    plt.figure()
    rates = [int(k.split("_")[-1]) for k in drop_keys]
    for ds in dataset_names:
        mai_vals = [experiment_data[k][ds]["mai"][-1] for k in drop_keys]
        plt.plot(rates, mai_vals, marker="o", label=ds)
    plt.xlabel("Token Dropout Rate (%)")
    plt.ylabel("Final Epoch MAI")
    plt.title("MAI vs Token Dropout Rate\nFinal Epoch MAI for Each Dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mai_vs_dropout_rate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MAI vs dropout plot: {e}")
    plt.close()
