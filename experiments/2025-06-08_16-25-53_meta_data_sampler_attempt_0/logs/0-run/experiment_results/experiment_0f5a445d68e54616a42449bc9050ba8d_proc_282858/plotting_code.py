import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    data_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for mode, ds_dict in experiment_data.items():
    for ds_name, exp in ds_dict.items():
        # Accuracy curve
        try:
            plt.figure()
            epochs = range(1, len(exp["metrics"]["train"]) + 1)
            plt.plot(epochs, exp["metrics"]["train"], label="Train")
            plt.plot(epochs, exp["metrics"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} ({mode}): Training vs Validation Accuracy")
            plt.legend()
            fname = f"{ds_name}_{mode}_accuracy_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating accuracy plot for {ds_name}-{mode}: {e}")
        finally:
            plt.close()
        # Loss curve
        try:
            plt.figure()
            epochs = range(1, len(exp["losses"]["train"]) + 1)
            plt.plot(epochs, exp["losses"]["train"], label="Train")
            plt.plot(epochs, exp["losses"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} ({mode}): Training vs Validation Loss")
            plt.legend()
            fname = f"{ds_name}_{mode}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}-{mode}: {e}")
        finally:
            plt.close()
        # Full-meta specific plots
        if mode == "full_meta":
            # Spearman correlations
            try:
                plt.figure()
                xs = range(1, len(exp["corrs"]) + 1)
                plt.plot(xs, exp["corrs"], marker="o")
                plt.xlabel("Meta-Update Step")
                plt.ylabel("Spearman Correlation")
                plt.title(f"{ds_name} ({mode}): Meta-Model Correlation History")
                fname = f"{ds_name}_{mode}_corr_history.png"
                plt.savefig(os.path.join(working_dir, fname))
            except Exception as e:
                print(f"Error creating corr plot for {ds_name}: {e}")
            finally:
                plt.close()
            # N_meta history
            try:
                plt.figure()
                xs = range(1, len(exp["N_meta_history"]) + 1)
                plt.plot(xs, exp["N_meta_history"], marker="o")
                plt.xlabel("Meta-Update Step")
                plt.ylabel("N_meta")
                plt.title(f"{ds_name} ({mode}): N_meta Adjustment History")
                fname = f"{ds_name}_{mode}_Nmeta_history.png"
                plt.savefig(os.path.join(working_dir, fname))
            except Exception as e:
                print(f"Error creating N_meta plot for {ds_name}: {e}")
            finally:
                plt.close()
