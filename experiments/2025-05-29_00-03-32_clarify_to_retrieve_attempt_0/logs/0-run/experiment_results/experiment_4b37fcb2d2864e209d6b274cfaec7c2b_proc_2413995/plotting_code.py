import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_data = {}

# Extract budgets and metrics
datasets = list(exp_data.get("clarification_turn_budget", {}).keys())
metrics = {
    name: exp_data["clarification_turn_budget"][name]["metrics"]["val"]
    for name in datasets
}
if datasets:
    budgets = [m["budget"] for m in metrics[datasets[0]]]
else:
    budgets = []

baseline_acc = {n: [m["baseline_acc"] for m in metrics[n]] for n in datasets}
clar_acc = {n: [m["clar_acc"] for m in metrics[n]] for n in datasets}
avg_turns = {n: [m["avg_turns"] for m in metrics[n]] for n in datasets}
ces_scores = {n: [m["CES"] for m in metrics[n]] for n in datasets}

# Plot 1: Validation Accuracies
try:
    plt.figure()
    for name in datasets:
        plt.plot(budgets, baseline_acc[name], marker="o", label=f"{name} Baseline")
        plt.plot(budgets, clar_acc[name], marker="x", label=f"{name} Clarified")
    plt.title("Budget vs Validation Accuracies")
    plt.suptitle("Baseline vs Clarified accuracy across datasets")
    plt.xlabel("Clarification Turn Budget")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_accuracy_vs_budget.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 2: Average Clarification Turns
try:
    plt.figure()
    for name in datasets:
        plt.plot(budgets, avg_turns[name], marker="o", label=name)
    plt.title("Budget vs Average Clarification Turns")
    plt.suptitle("Average number of clarification turns used per sample")
    plt.xlabel("Clarification Turn Budget")
    plt.ylabel("Average Turns")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_avg_turns_vs_budget.png"))
    plt.close()
except Exception as e:
    print(f"Error creating avg turns plot: {e}")
    plt.close()

# Plot 3: Clarification Efficiency Score (CES)
try:
    plt.figure()
    for name in datasets:
        plt.plot(budgets, ces_scores[name], marker="o", label=name)
    plt.title("Budget vs Clarification Efficiency Score (CES)")
    plt.suptitle("Clarification Efficiency Score across datasets")
    plt.xlabel("Clarification Turn Budget")
    plt.ylabel("CES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_ces_vs_budget.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES plot: {e}")
    plt.close()
