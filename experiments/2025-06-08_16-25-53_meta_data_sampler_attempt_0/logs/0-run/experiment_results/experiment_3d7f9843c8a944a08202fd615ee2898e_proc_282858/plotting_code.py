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
else:
    for ablation, ds_dict in experiment_data.items():
        for ds_name, stats in ds_dict.items():
            # Accuracy curves
            try:
                tr_acc = stats["metrics"]["train"]
                val_acc = stats["metrics"]["val"]
                plt.figure()
                plt.plot(tr_acc, label="Train Acc")
                plt.plot(val_acc, label="Val Acc")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"{ds_name} Accuracy Curve\nTrain vs Validation Accuracy")
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curve.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating {ds_name} accuracy curve: {e}")
                plt.close()
            # Loss curves
            try:
                tr_loss = stats["losses"]["train"]
                val_loss = stats["losses"]["val"]
                plt.figure()
                plt.plot(tr_loss, label="Train Loss")
                plt.plot(val_loss, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{ds_name} Loss Curve\nTrain vs Validation Loss")
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating {ds_name} loss curve: {e}")
                plt.close()
            # Spearman correlation history
            try:
                corrs = stats.get("corrs", [])
                if corrs:
                    plt.figure()
                    plt.plot(corrs, marker="o")
                    plt.xlabel("Meta-update Step")
                    plt.ylabel("Spearman Correlation")
                    plt.title(f"{ds_name} Spearman Correlation\nper Meta-update")
                    plt.savefig(
                        os.path.join(working_dir, f"{ds_name}_spearman_corr.png")
                    )
                    plt.close()
            except Exception as e:
                print(f"Error creating {ds_name} spearman correlation plot: {e}")
                plt.close()
            # N_meta history
            try:
                n_meta = stats.get("N_meta_history", [])
                if n_meta:
                    plt.figure()
                    plt.plot(n_meta, marker="o")
                    plt.xlabel("Meta-update Step")
                    plt.ylabel("N_meta Value")
                    plt.title(f"{ds_name} N_meta History\nper Meta-update")
                    plt.savefig(
                        os.path.join(working_dir, f"{ds_name}_N_meta_history.png")
                    )
                    plt.close()
            except Exception as e:
                print(f"Error creating {ds_name} N_meta history plot: {e}")
                plt.close()
