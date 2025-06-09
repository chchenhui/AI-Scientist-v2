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

datasets = list(experiment_data["single"].keys())

try:
    plt.figure()
    idx = np.arange(len(datasets))
    width = 0.35
    base = [experiment_data["single"][d]["metrics"]["val"][0] for d in datasets]
    clar = [experiment_data["single"][d]["metrics"]["val"][1] for d in datasets]
    plt.bar(idx - width / 2, base, width, label="Baseline")
    plt.bar(idx + width / 2, clar, width, label="Clarification")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("Single Clarification: Accuracy Comparison")
    plt.suptitle("Left: Baseline, Right: Clarification")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "all_datasets_single_accuracy_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating single accuracy plot: {e}")
    plt.close()

try:
    plt.figure()
    idx = np.arange(len(datasets))
    base = [experiment_data["iterative"][d]["metrics"]["val"][0] for d in datasets]
    clar = [experiment_data["iterative"][d]["metrics"]["val"][1] for d in datasets]
    plt.bar(idx - width / 2, base, width, label="Baseline")
    plt.bar(idx + width / 2, clar, width, label="Clarification")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("Iterative Clarification: Accuracy Comparison")
    plt.suptitle("Left: Baseline, Right: Clarification")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "all_datasets_iterative_accuracy_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating iterative accuracy plot: {e}")
    plt.close()

try:
    plt.figure()
    idx = np.arange(len(datasets))
    ces1 = [experiment_data["single"][d]["metrics"]["val"][3] for d in datasets]
    ces2 = [experiment_data["iterative"][d]["metrics"]["val"][3] for d in datasets]
    plt.bar(idx - width / 2, ces1, width, label="Single")
    plt.bar(idx + width / 2, ces2, width, label="Iterative")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("CES")
    plt.title("Cost-Effectiveness Score Comparison")
    plt.suptitle("Left: Single, Right: Iterative")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_ces_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES comparison plot: {e}")
    plt.close()

try:
    plt.figure()
    idx = np.arange(len(datasets))
    t1 = [experiment_data["single"][d]["metrics"]["val"][2] for d in datasets]
    t2 = [experiment_data["iterative"][d]["metrics"]["val"][2] for d in datasets]
    plt.bar(idx - width / 2, t1, width, label="Single")
    plt.bar(idx + width / 2, t2, width, label="Iterative")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("Average Turns")
    plt.title("Average Number of Clarification Turns")
    plt.suptitle("Left: Single, Right: Iterative")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_average_turns_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating average turns plot: {e}")
    plt.close()

for ab in ["single", "iterative"]:
    for d in datasets:
        v = experiment_data[ab][d]["metrics"]["val"]
        print(
            f"{ab.upper()} {d}: baseline_acc={v[0]:.4f}, clar_acc={v[1]:.4f}, avg_turns={v[2]:.4f}, CES={v[3]:.4f}"
        )
