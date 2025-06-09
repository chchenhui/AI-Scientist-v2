import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

flip_rates = experiment_data.get("ambiguity_detection_noise", {}).get("flip_rates", [])
datasets = ["SQuAD", "AmbigQA", "TriviaQA-rc"]
metrics = ["baseline_acc", "clar_acc", "avg_turns", "CES"]

for metric in metrics:
    try:
        plt.figure()
        for ds in datasets:
            values = experiment_data["ambiguity_detection_noise"][ds]["metrics"][metric]
            plt.plot(flip_rates, values, marker="o", label=ds)
        plt.title(
            f"{metric.replace('_', ' ').title()} vs Flip Rate\nDatasets: SQuAD, AmbigQA, TriviaQA-rc"
        )
        plt.xlabel("Flip Rate")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        fname = f"fliprate_{metric}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {metric}: {e}")
        plt.close()
