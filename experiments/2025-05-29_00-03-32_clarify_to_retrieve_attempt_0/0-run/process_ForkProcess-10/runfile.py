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
    experiment_data = {}

metrics = experiment_data.get("metrics", {})
names = list(metrics.keys())
baseline_acc = [metrics[n]["baseline_acc"] for n in names]
clar_acc = [metrics[n]["clar_acc"] for n in names]
ces_scores = [metrics[n]["CES"] for n in names]
avg_turns = [metrics[n]["avg_turns"] for n in names]

print("Datasets:", names)
print("Baseline Accuracies:", baseline_acc)
print("Clarification Accuracies:", clar_acc)
print("CES Scores:", ces_scores)
print("Average Turns:", avg_turns)

# Plot 1: Baseline vs Clarification Accuracy
try:
    plt.figure()
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width / 2, baseline_acc, width, label="Baseline Acc")
    plt.bar(x + width / 2, clar_acc, width, label="Clarification Acc")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("QA Datasets: Baseline vs Clarification Accuracy")
    plt.xticks(x, names)
    plt.legend()
    plt.savefig(os.path.join(working_dir, "qa_accuracy_comparison_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy comparison plot: {e}")
    plt.close()

# Plot 2: CES Comparison
try:
    plt.figure()
    plt.bar(names, ces_scores)
    plt.xlabel("Dataset")
    plt.ylabel("Clarification Efficiency Score (CES)")
    plt.title("QA Datasets: CES Comparison")
    plt.savefig(os.path.join(working_dir, "qa_ces_comparison_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES comparison plot: {e}")
    plt.close()

# Plot 3: Average Clarification Turns
try:
    plt.figure()
    plt.bar(names, avg_turns)
    plt.xlabel("Dataset")
    plt.ylabel("Average Number of Clarification Turns")
    plt.title("QA Datasets: Average Clarification Turns")
    plt.savefig(os.path.join(working_dir, "qa_avg_turns_comparison_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating average turns plot: {e}")
    plt.close()
