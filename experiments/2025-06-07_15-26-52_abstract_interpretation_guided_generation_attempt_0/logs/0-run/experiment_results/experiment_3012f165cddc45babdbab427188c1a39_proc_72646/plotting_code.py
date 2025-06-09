import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    baseline = data["metrics"]["baseline"]
    guided = data["metrics"]["guided"]
    tasks = [d["task"] for d in data.get("generated_code", [])]

    # Plot 1: Overall error-free rates
    try:
        rate0 = sum(baseline) / len(baseline)
        rate1 = sum(guided) / len(guided)
        plt.figure()
        plt.bar(["Baseline", "Guided"], [rate0, rate1], color=["red", "green"])
        plt.ylabel("Error-Free Generation Rate")
        plt.title(
            "Error-Free Generation Rate on Synthetic Tasks\nDataset: Synthetic Arithmetic Tasks"
        )
        plt.savefig(
            os.path.join(working_dir, "synthetic_tasks_error_rate_comparison.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating overall error rate plot: {e}")
        plt.close()

    # Plot 2: Per-task correctness
    try:
        x = np.arange(len(tasks))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, baseline, width, label="Baseline", color="red")
        plt.bar(x + width / 2, guided, width, label="Guided", color="green")
        plt.xticks(x, tasks)
        plt.ylabel("Error-Free Outcome (1=Success)")
        plt.title("Per-Task Error-Free Outcome\nDataset: Synthetic Arithmetic Tasks")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_tasks_per_task_correctness.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating per-task correctness plot: {e}")
        plt.close()
