import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_paths = [
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_d5443f05c1dd4b70a7ea4fc3372f0f91_proc_2404624/experiment_data.npy",
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_eb8519ec2e8748f8a7c958f78fd99f3b_proc_2404625/experiment_data.npy",
    "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_af9f608fb5ca422788fa1880de667a83_proc_2404623/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
try:
    for p in experiment_data_paths:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), p)
        ed = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate metrics across runs
metrics_by_dataset = {}
for ed in all_experiment_data:
    for ds, m in ed.get("metrics", {}).items():
        if ds not in metrics_by_dataset:
            metrics_by_dataset[ds] = {
                "baseline": [],
                "clar": [],
                "ces": [],
                "turns": [],
            }
        metrics_by_dataset[ds]["baseline"].append(m.get("baseline_acc", np.nan))
        metrics_by_dataset[ds]["clar"].append(m.get("clar_acc", np.nan))
        metrics_by_dataset[ds]["ces"].append(m.get("CES", np.nan))
        metrics_by_dataset[ds]["turns"].append(m.get("avg_turns", np.nan))

names = sorted(metrics_by_dataset.keys())
baseline_means, baseline_sems = [], []
clar_means, clar_sems = [], []
ces_means, ces_sems = [], []
turns_means, turns_sems = [], []

for ds in names:
    b = np.array(metrics_by_dataset[ds]["baseline"], dtype=float)
    c = np.array(metrics_by_dataset[ds]["clar"], dtype=float)
    cs = np.array(metrics_by_dataset[ds]["ces"], dtype=float)
    t = np.array(metrics_by_dataset[ds]["turns"], dtype=float)
    n = len(b)
    baseline_means.append(b.mean() if n > 0 else np.nan)
    baseline_sems.append(b.std(ddof=1) / np.sqrt(n) if n > 1 else 0)
    clar_means.append(c.mean() if n > 0 else np.nan)
    clar_sems.append(c.std(ddof=1) / np.sqrt(n) if n > 1 else 0)
    ces_means.append(cs.mean() if n > 0 else np.nan)
    ces_sems.append(cs.std(ddof=1) / np.sqrt(n) if n > 1 else 0)
    turns_means.append(t.mean() if n > 0 else np.nan)
    turns_sems.append(t.std(ddof=1) / np.sqrt(n) if n > 1 else 0)

# Print aggregated metrics
print("Datasets:", names)
print("Baseline Acc (mean ± SEM):", list(zip(baseline_means, baseline_sems)))
print("Clarification Acc (mean ± SEM):", list(zip(clar_means, clar_sems)))
print("CES (mean ± SEM):", list(zip(ces_means, ces_sems)))
print("Avg Clarification Turns (mean ± SEM):", list(zip(turns_means, turns_sems)))

# Plot 1: Baseline vs Clarification Accuracy
try:
    plt.figure()
    x = np.arange(len(names))
    width = 0.35
    plt.bar(
        x - width / 2,
        baseline_means,
        width,
        yerr=baseline_sems,
        capsize=5,
        label="Baseline Acc",
    )
    plt.bar(
        x + width / 2,
        clar_means,
        width,
        yerr=clar_sems,
        capsize=5,
        label="Clarification Acc",
    )
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title(
        "QA Datasets: Baseline vs Clarification Accuracy\n(Error bars: Standard Error)"
    )
    plt.xticks(x, names)
    plt.legend(title="Metric")
    plt.savefig(os.path.join(working_dir, "qa_baseline_vs_clar_accuracy_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy comparison plot: {e}")
    plt.close()

# Plot 2: CES Comparison
try:
    plt.figure()
    plt.bar(names, ces_means, yerr=ces_sems, capsize=5, color="skyblue")
    plt.xlabel("Dataset")
    plt.ylabel("Clarification Efficiency Score (CES)")
    plt.title("QA Datasets: CES Comparison\n(Error bars: Standard Error)")
    plt.savefig(os.path.join(working_dir, "qa_ces_comparison_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES comparison plot: {e}")
    plt.close()

# Plot 3: Average Clarification Turns
try:
    plt.figure()
    plt.bar(names, turns_means, yerr=turns_sems, capsize=5, color="lightgreen")
    plt.xlabel("Dataset")
    plt.ylabel("Average Clarification Turns")
    plt.title("QA Datasets: Average Clarification Turns\n(Error bars: Standard Error)")
    plt.savefig(os.path.join(working_dir, "qa_avg_turns_comparison_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating average turns plot: {e}")
    plt.close()

# Plot 4: Training/Validation Loss Curves if available
try:
    if all("train_loss" in ed and "val_loss" in ed for ed in all_experiment_data):
        lengths = [len(ed["train_loss"]) for ed in all_experiment_data]
        min_len = min(lengths)
        train_arr = np.array([ed["train_loss"][:min_len] for ed in all_experiment_data])
        val_arr = np.array([ed["val_loss"][:min_len] for ed in all_experiment_data])
        epochs = np.arange(1, min_len + 1)
        m_train = train_arr.mean(axis=0)
        s_train = train_arr.std(axis=0, ddof=1) / np.sqrt(train_arr.shape[0])
        m_val = val_arr.mean(axis=0)
        s_val = val_arr.std(axis=0, ddof=1) / np.sqrt(val_arr.shape[0])
        plt.figure()
        plt.errorbar(epochs, m_train, yerr=s_train, capsize=3, label="Train Loss")
        plt.errorbar(epochs, m_val, yerr=s_val, capsize=3, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("QA Datasets: Training vs Validation Loss\n(Mean ± Standard Error)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "qa_train_val_loss_curves_mean_sem.png"))
        plt.close()
except Exception as e:
    print(f"Error creating train/validation curves plot: {e}")
    plt.close()
