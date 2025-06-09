import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = data["Ablate_Label_Noise_Robustness"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# Generate plots
for ds_name, ds_res in exp.items():
    for noise, res in ds_res.items():
        # Extract arrays
        train_acc = res["metrics"]["train"]
        val_acc = res["metrics"]["val"]
        train_loss = res["losses"]["train"]
        val_loss = res["losses"]["val"]
        corrs = res.get("corrs", [])
        n_meta = res.get("N_meta_history", [])

        # Accuracy curves
        try:
            plt.figure()
            plt.plot(train_acc, label="Train Accuracy")
            plt.plot(val_acc, label="Validation Accuracy")
            plt.title(f"Accuracy Curves - {ds_name} ({noise}% noise)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_{noise}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {ds_name} {noise}%: {e}")
            plt.close()

        # Loss curves
        try:
            plt.figure()
            plt.plot(train_loss, label="Train Loss")
            plt.plot(val_loss, label="Validation Loss")
            plt.title(f"Loss Curves - {ds_name} ({noise}% noise)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_{noise}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name} {noise}%: {e}")
            plt.close()

        # Spearman correlation history
        try:
            plt.figure()
            plt.plot(corrs, marker="o")
            plt.title(f"Spearman Corr History - {ds_name} ({noise}% noise)")
            plt.xlabel("Meta-update Step")
            plt.ylabel("Spearman Correlation")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_corr_{noise}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating corr history plot for {ds_name} {noise}%: {e}")
            plt.close()

        # N_meta history
        try:
            plt.figure()
            plt.plot(n_meta, marker="o")
            plt.title(f"N_meta History - {ds_name} ({noise}% noise)")
            plt.xlabel("Meta-update Step")
            plt.ylabel("N_meta")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_Nmeta_{noise}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating N_meta history plot for {ds_name} {noise}%: {e}")
            plt.close()
