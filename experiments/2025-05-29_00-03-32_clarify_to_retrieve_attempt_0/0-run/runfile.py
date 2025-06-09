import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_25802c7b675646a7bc7836e81c89b0bd_proc_2413997/experiment_data.npy",
        "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_07957a36ecb844f6ab77bb29e1c87c2e_proc_2413996/experiment_data.npy",
        "experiments/2025-05-29_00-03-32_clarify_to_retrieve_attempt_0/logs/0-run/experiment_results/experiment_3e3ab5e21b1b4e14923cd7e8040831f8_proc_2413995/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT"), p)
        exp = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate metrics
datasets = list(all_experiment_data[0]["single"].keys())
n_reps = len(all_experiment_data)
data_agg = {"single": {}, "iterative": {}}
for ab in ["single", "iterative"]:
    for d in datasets:
        reps = np.array([exp[ab][d]["metrics"]["val"] for exp in all_experiment_data])
        baseline, clar, turns, ces = reps[:, 0], reps[:, 1], reps[:, 2], reps[:, 3]
        data_agg[ab][d] = {
            "baseline_mean": baseline.mean(),
            "baseline_sem": baseline.std() / np.sqrt(n_reps),
            "clar_mean": clar.mean(),
            "clar_sem": clar.std() / np.sqrt(n_reps),
            "turns_mean": turns.mean(),
            "turns_sem": turns.std() / np.sqrt(n_reps),
            "ces_mean": ces.mean(),
            "ces_sem": ces.std() / np.sqrt(n_reps),
        }

idx = np.arange(len(datasets))
width = 0.35

# Plot aggregated single accuracy
try:
    plt.figure()
    b_means = np.array([data_agg["single"][d]["baseline_mean"] for d in datasets])
    b_sems = np.array([data_agg["single"][d]["baseline_sem"] for d in datasets])
    c_means = np.array([data_agg["single"][d]["clar_mean"] for d in datasets])
    c_sems = np.array([data_agg["single"][d]["clar_sem"] for d in datasets])
    plt.bar(idx - width / 2, b_means, width, yerr=b_sems, label="Baseline")
    plt.bar(idx + width / 2, c_means, width, yerr=c_sems, label="Clarification")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("Aggregated Single Clarification: Accuracy Comparison")
    plt.suptitle("Left: Baseline, Right: Clarification")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_single_accuracy_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated single accuracy plot: {e}")
    plt.close()

# Plot aggregated iterative accuracy
try:
    plt.figure()
    b_means = np.array([data_agg["iterative"][d]["baseline_mean"] for d in datasets])
    b_sems = np.array([data_agg["iterative"][d]["baseline_sem"] for d in datasets])
    c_means = np.array([data_agg["iterative"][d]["clar_mean"] for d in datasets])
    c_sems = np.array([data_agg["iterative"][d]["clar_sem"] for d in datasets])
    plt.bar(idx - width / 2, b_means, width, yerr=b_sems, label="Baseline")
    plt.bar(idx + width / 2, c_means, width, yerr=c_sems, label="Clarification")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title("Aggregated Iterative Clarification: Accuracy Comparison")
    plt.suptitle("Left: Baseline, Right: Clarification")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "aggregated_iterative_accuracy_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated iterative accuracy plot: {e}")
    plt.close()

# Plot aggregated CES comparison
try:
    plt.figure()
    s_means = np.array([data_agg["single"][d]["ces_mean"] for d in datasets])
    s_sems = np.array([data_agg["single"][d]["ces_sem"] for d in datasets])
    i_means = np.array([data_agg["iterative"][d]["ces_mean"] for d in datasets])
    i_sems = np.array([data_agg["iterative"][d]["ces_sem"] for d in datasets])
    plt.bar(idx - width / 2, s_means, width, yerr=s_sems, label="Single")
    plt.bar(idx + width / 2, i_means, width, yerr=i_sems, label="Iterative")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("CES")
    plt.title("Aggregated Cost-Effectiveness Score Comparison")
    plt.suptitle("Left: Single, Right: Iterative")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_ces_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CES comparison plot: {e}")
    plt.close()

# Plot aggregated average turns
try:
    plt.figure()
    s_means = np.array([data_agg["single"][d]["turns_mean"] for d in datasets])
    s_sems = np.array([data_agg["single"][d]["turns_sem"] for d in datasets])
    i_means = np.array([data_agg["iterative"][d]["turns_mean"] for d in datasets])
    i_sems = np.array([data_agg["iterative"][d]["turns_sem"] for d in datasets])
    plt.bar(idx - width / 2, s_means, width, yerr=s_sems, label="Single")
    plt.bar(idx + width / 2, i_means, width, yerr=i_sems, label="Iterative")
    plt.xticks(idx, datasets)
    plt.xlabel("Dataset")
    plt.ylabel("Average Turns")
    plt.title("Aggregated Average Number of Clarification Turns")
    plt.suptitle("Left: Single, Right: Iterative")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_average_turns_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated average turns plot: {e}")
    plt.close()

# Plot training/validation curves if available
try:
    first = all_experiment_data[0]
    if "train" in first and "val" in first:
        train_curves = np.array([exp["train"] for exp in all_experiment_data])
        val_curves = np.array([exp["val"] for exp in all_experiment_data])
        t_mean = train_curves.mean(axis=0)
        t_sem = train_curves.std(axis=0) / np.sqrt(n_reps)
        v_mean = val_curves.mean(axis=0)
        v_sem = val_curves.std(axis=0) / np.sqrt(n_reps)
        epochs = np.arange(len(t_mean))
        plt.figure()
        plt.errorbar(epochs, t_mean, yerr=t_sem, label="Train")
        plt.errorbar(epochs, v_mean, yerr=v_sem, label="Validation")
        plt.title("Aggregated Training and Validation Curves")
        plt.suptitle("Mean ± SEM across experiments")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "aggregated_train_val_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating training/validation curves: {e}")
    plt.close()

# Print aggregated metrics
for ab in ["single", "iterative"]:
    for d in datasets:
        agg = data_agg[ab][d]
        print(
            f"{ab.upper()} {d}: "
            f"baseline_acc={agg['baseline_mean']:.4f}±{agg['baseline_sem']:.4f}, "
            f"clar_acc={agg['clar_mean']:.4f}±{agg['clar_sem']:.4f}, "
            f"avg_turns={agg['turns_mean']:.4f}±{agg['turns_sem']:.4f}, "
            f"CES={agg['ces_mean']:.4f}±{agg['ces_sem']:.4f}"
        )
