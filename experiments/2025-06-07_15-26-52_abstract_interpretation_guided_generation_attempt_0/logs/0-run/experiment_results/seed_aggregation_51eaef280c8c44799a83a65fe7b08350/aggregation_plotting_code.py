import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load multiple experiment_data files
experiment_data_path_list = [
    "experiments/2025-06-07_15-26-52_abstract_interpretation_guided_generation_attempt_0/logs/0-run/experiment_results/experiment_79fce44eb3104f6bbd598a51a2a63e0f_proc_72647/experiment_data.npy",
    "experiments/2025-06-07_15-26-52_abstract_interpretation_guided_generation_attempt_0/logs/0-run/experiment_results/experiment_ab8777ef1bc746bc80f9852bcfdaa275_proc_72646/experiment_data.npy",
]

all_experiment_data = []
for path in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data from {path}: {e}")

if not all_experiment_data:
    print("No data loaded, exiting.")
else:
    # Assume all runs share same dataset keys
    for dataset_name in all_experiment_data[0].keys():
        # Collect losses and metrics across runs
        losses_train = []
        losses_val = []
        rates_train = []
        rates_val = []
        for run in all_experiment_data:
            ds = run.get(dataset_name, {})
            l = ds.get("losses", {})
            m = ds.get("metrics", {})
            if "train" in l:
                losses_train.append(l["train"])
            if "val" in l:
                losses_val.append(l["val"])
            if "train" in m:
                rates_train.append(m["train"])
            if "val" in m:
                rates_val.append(m["val"])

        # Convert to arrays if we have at least one run
        if losses_train:
            lt = np.vstack([np.array(x) for x in losses_train])
            mean_lt = lt.mean(axis=0)
            sem_lt = lt.std(axis=0, ddof=1) / np.sqrt(lt.shape[0])
            epochs = np.arange(1, mean_lt.size + 1)

            try:
                plt.figure()
                plt.plot(epochs, mean_lt, label="Mean Train Loss")
                plt.fill_between(
                    epochs,
                    mean_lt - sem_lt,
                    mean_lt + sem_lt,
                    alpha=0.3,
                    label="SEM Train Loss",
                )
                if losses_val:
                    lv = np.vstack([np.array(x) for x in losses_val])
                    mean_lv = lv.mean(axis=0)
                    sem_lv = lv.std(axis=0, ddof=1) / np.sqrt(lv.shape[0])
                    plt.plot(epochs, mean_lv, label="Mean Val Loss")
                    plt.fill_between(
                        epochs,
                        mean_lv - sem_lv,
                        mean_lv + sem_lv,
                        alpha=0.3,
                        label="SEM Val Loss",
                    )
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{dataset_name} Dataset Loss Curve\n(Aggregated Mean ± SEM)")
                plt.legend()
                fname = f"{dataset_name}_loss_mean_sem.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating loss curve plot for {dataset_name}: {e}")
                plt.close()

        if rates_train:
            rt = np.vstack([np.array(x) for x in rates_train])
            mean_rt = rt.mean(axis=0)
            sem_rt = rt.std(axis=0, ddof=1) / np.sqrt(rt.shape[0])
            epochs = np.arange(1, mean_rt.size + 1)

            try:
                plt.figure()
                plt.plot(epochs, mean_rt, label="Mean Train Error-Free Rate")
                plt.fill_between(
                    epochs,
                    mean_rt - sem_rt,
                    mean_rt + sem_rt,
                    alpha=0.3,
                    label="SEM Train Rate",
                )
                if rates_val:
                    rv = np.vstack([np.array(x) for x in rates_val])
                    mean_rv = rv.mean(axis=0)
                    sem_rv = rv.std(axis=0, ddof=1) / np.sqrt(rv.shape[0])
                    plt.plot(epochs, mean_rv, label="Mean Val Error-Free Rate")
                    plt.fill_between(
                        epochs,
                        mean_rv - sem_rv,
                        mean_rv + sem_rv,
                        alpha=0.3,
                        label="SEM Val Rate",
                    )
                plt.xlabel("Epoch")
                plt.ylabel("Error-Free Generation Rate")
                plt.title(
                    f"{dataset_name} Dataset Error-Free Rate\n(Aggregated Mean ± SEM)"
                )
                plt.legend()
                fname = f"{dataset_name}_error_rate_mean_sem.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating error rate plot for {dataset_name}: {e}")
                plt.close()
