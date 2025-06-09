import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_f9dfbfebf907482d8cfa805ed5d250f5_proc_119935/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_57ac4e99a24644f9b02c4748160b834a_proc_119936/experiment_data.npy",
    "experiments/2025-06-07_19-09-39_weight_primitives_attempt_0/logs/0-run/experiment_results/experiment_9f9d63180c0b46548dd5bd69d0b77887_proc_119934/experiment_data.npy",
]

# Load all experiment data
all_experiment_data = []
for exp_path in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), exp_path), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data: {e}")

# Extract synthetic dataset runs
alt_data_list = [
    d.get("alt_min_freq", {}).get("synthetic", {}) for d in all_experiment_data
]
if not alt_data_list:
    print("No synthetic data found.")
else:
    ratios = alt_data_list[0].get("ratios", [])
    n_reps = len(alt_data_list)
    metrics_train_list = [d.get("metrics", {}).get("train", []) for d in alt_data_list]
    metrics_val_list = [d.get("metrics", {}).get("val", []) for d in alt_data_list]
    losses_train_list = [d.get("losses", {}).get("train", []) for d in alt_data_list]
    losses_val_list = [d.get("losses", {}).get("val", []) for d in alt_data_list]

    # Compute mean and SEM curves for error
    mean_train_curves, sem_train_curves = [], []
    mean_val_curves, sem_val_curves = [], []
    for i in range(len(ratios)):
        # Align by minimum epochs across reps
        len_tr = [len(rep[i]) for rep in metrics_train_list if len(rep) > i]
        min_tr = min(len_tr) if len_tr else 0
        arr_tr = np.array([rep[i][:min_tr] for rep in metrics_train_list])
        mean_tr = arr_tr.mean(axis=0)
        sem_tr = arr_tr.std(axis=0, ddof=1) / np.sqrt(n_reps)
        len_vl = [len(rep[i]) for rep in metrics_val_list if len(rep) > i]
        min_vl = min(len_vl) if len_vl else 0
        arr_vl = np.array([rep[i][:min_vl] for rep in metrics_val_list])
        mean_vl = arr_vl.mean(axis=0)
        sem_vl = arr_vl.std(axis=0, ddof=1) / np.sqrt(n_reps)
        mean_train_curves.append(mean_tr)
        sem_train_curves.append(sem_tr)
        mean_val_curves.append(mean_vl)
        sem_val_curves.append(sem_vl)

    # Aggregated error plot
    try:
        plt.figure()
        for (c, d), mean_tr, sem_tr, mean_vl, sem_vl in zip(
            ratios, mean_train_curves, sem_train_curves, mean_val_curves, sem_val_curves
        ):
            epochs_tr = range(1, len(mean_tr) + 1)
            epochs_vl = range(1, len(mean_vl) + 1)
            plt.plot(epochs_tr, mean_tr, label=f"{c}:{d} Train Mean", linestyle="-")
            plt.fill_between(epochs_tr, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
            plt.plot(epochs_vl, mean_vl, label=f"{c}:{d} Val Mean", linestyle="--")
            plt.fill_between(epochs_vl, mean_vl - sem_vl, mean_vl + sem_vl, alpha=0.2)
        plt.title("Synthetic Dataset: Aggregated Training and Validation Error")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.legend(title="Mean ± SEM")
        plt.suptitle("Solid: Train Mean, Dashed: Val Mean; Shaded: ± SEM", fontsize=10)
        plt.savefig(
            os.path.join(working_dir, "synthetic_aggregated_train_val_error.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated error plot: {e}")
        plt.close()

    # Compute and plot aggregated loss
    try:
        mean_loss_train, sem_loss_train = [], []
        mean_loss_val, sem_loss_val = [], []
        for i in range(len(ratios)):
            lt = [len(rep[i]) for rep in losses_train_list if len(rep) > i]
            min_lt = min(lt) if lt else 0
            arr_lt = np.array([rep[i][:min_lt] for rep in losses_train_list])
            m_lt = arr_lt.mean(axis=0)
            s_lt = arr_lt.std(axis=0, ddof=1) / np.sqrt(n_reps)
            lv = [len(rep[i]) for rep in losses_val_list if len(rep) > i]
            min_lv = min(lv) if lv else 0
            arr_lv = np.array([rep[i][:min_lv] for rep in losses_val_list])
            m_lv = arr_lv.mean(axis=0)
            s_lv = arr_lv.std(axis=0, ddof=1) / np.sqrt(n_reps)
            mean_loss_train.append(m_lt)
            sem_loss_train.append(s_lt)
            mean_loss_val.append(m_lv)
            sem_loss_val.append(s_lv)
        plt.figure()
        for (c, d), m_lt, s_lt, m_lv, s_lv in zip(
            ratios, mean_loss_train, sem_loss_train, mean_loss_val, sem_loss_val
        ):
            e_lt = range(1, len(m_lt) + 1)
            e_lv = range(1, len(m_lv) + 1)
            plt.plot(e_lt, m_lt, label=f"{c}:{d} Train Loss Mean", linestyle="-")
            plt.fill_between(e_lt, m_lt - s_lt, m_lt + s_lt, alpha=0.2)
            plt.plot(e_lv, m_lv, label=f"{c}:{d} Val Loss Mean", linestyle="--")
            plt.fill_between(e_lv, m_lv - s_lv, m_lv + s_lv, alpha=0.2)
        plt.title("Synthetic Dataset: Aggregated Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend(title="Mean ± SEM")
        plt.suptitle("Solid: Train Mean, Dashed: Val Mean; Shaded: ± SEM", fontsize=10)
        plt.savefig(
            os.path.join(working_dir, "synthetic_aggregated_train_val_loss.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # Print aggregated final validation errors
    try:
        print("Aggregated final validation errors per ratio:")
        for i, (c, d) in enumerate(ratios):
            finals = [rep[i][-1] for rep in metrics_val_list if len(rep) > i]
            if finals:
                m = np.mean(finals)
                se = np.std(finals, ddof=1) / np.sqrt(n_reps)
                print(f"Ratio {c}:{d} -> {m:.4f} ± {se:.4f}")
    except Exception as e:
        print(f"Error printing final validation errors: {e}")
