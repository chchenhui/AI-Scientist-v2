import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load all experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_d8ba5ee3c5a44f94b002565339e07c2a_proc_2561565/experiment_data.npy",
        "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_0c76e6b5896d4343a17fd7ba6713a6fb_proc_2561566/experiment_data.npy",
        "experiments/2025-05-29_15-59-15_coadaptive_explanation_alignment_attempt_0/logs/0-run/experiment_results/experiment_2ad53f5eba634a988a98ec655763bc43_proc_2561564/experiment_data.npy",
    ]
    all_experiment_data = []
    for exp_path in experiment_data_path_list:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), exp_path), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Aggregated accuracy curves
try:
    train_list, val_list = [], []
    for exp in all_experiment_data:
        static = exp.get("static_explainer", {})
        metrics = static.get("metrics", {})
        train_list.append(metrics.get("train", []))
        val_list.append(metrics.get("val", []))
    # Align lengths
    min_epochs = min(min(len(a) for a in train_list), min(len(a) for a in val_list))
    train_arr = np.array([a[:min_epochs] for a in train_list])
    val_arr = np.array([a[:min_epochs] for a in val_list])
    epochs = np.arange(1, min_epochs + 1)
    mean_train = train_arr.mean(axis=0)
    se_train = train_arr.std(axis=0, ddof=1) / np.sqrt(len(train_arr))
    mean_val = val_arr.mean(axis=0)
    se_val = val_arr.std(axis=0, ddof=1) / np.sqrt(len(val_arr))
    plt.figure()
    plt.errorbar(epochs, mean_train, yerr=se_train, label="Train")
    plt.errorbar(epochs, mean_val, yerr=se_val, label="Val")
    plt.title(
        "Static Explainer Accuracy Curves (Aggregated)\nTraining vs Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "static_explainer_accuracy_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# Aggregated loss curves
try:
    loss_train_list, loss_val_list = [], []
    for exp in all_experiment_data:
        static = exp.get("static_explainer", {})
        losses = static.get("losses", {})
        loss_train_list.append(losses.get("train", []))
        loss_val_list.append(losses.get("val", []))
    min_epochs = min(
        min(len(a) for a in loss_train_list), min(len(a) for a in loss_val_list)
    )
    lt = np.array([a[:min_epochs] for a in loss_train_list])
    lv = np.array([a[:min_epochs] for a in loss_val_list])
    epochs = np.arange(1, min_epochs + 1)
    mean_lt = lt.mean(axis=0)
    se_lt = lt.std(axis=0, ddof=1) / np.sqrt(len(lt))
    mean_lv = lv.mean(axis=0)
    se_lv = lv.std(axis=0, ddof=1) / np.sqrt(len(lv))
    plt.figure()
    plt.errorbar(epochs, mean_lt, yerr=se_lt, label="Train")
    plt.errorbar(epochs, mean_lv, yerr=se_lv, label="Val")
    plt.title("Static Explainer Loss Curves (Aggregated)\nTraining vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "static_explainer_loss_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# Aggregated test distribution
try:
    pred_counts, gt_counts = [], []
    for exp in all_experiment_data:
        static = exp.get("static_explainer", {})
        preds = static.get("predictions", [])
        gt = static.get("ground_truth", [])
        if len(preds) and len(gt):
            n_cls = max(max(preds), max(gt)) + 1
            pred_counts.append(np.bincount(preds, minlength=n_cls))
            gt_counts.append(np.bincount(gt, minlength=n_cls))
    if pred_counts:
        pc = np.array(pred_counts)
        gc = np.array(gt_counts)
        classes = np.arange(pc.shape[1])
        mean_pc = pc.mean(axis=0)
        se_pc = pc.std(axis=0, ddof=1) / np.sqrt(len(pc))
        mean_gc = gc.mean(axis=0)
        se_gc = gc.std(axis=0, ddof=1) / np.sqrt(len(gc))
        width = 0.35
        plt.figure()
        plt.bar(classes - width / 2, mean_gc, width, yerr=se_gc, label="Ground Truth")
        plt.bar(classes + width / 2, mean_pc, width, yerr=se_pc, label="Predictions")
        plt.title(
            "Static Explainer Test Predictions vs Ground Truth (Aggregated)\nTest Set Class Distribution (GT left, Pred right)"
        )
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(classes)
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, "static_explainer_test_distribution_aggregated.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated distribution plot: {e}")
    plt.close()
