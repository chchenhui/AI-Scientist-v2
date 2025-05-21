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
        for dataset_name, ed in ds_dict.items():
            epochs = list(range(1, len(ed["losses"]["train"]) + 1))
            try:
                plt.figure()
                plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
                plt.plot(epochs, ed["losses"]["val"], label="Validation Loss")
                plt.title(
                    f"Dataset {dataset_name} - Ablation {ablation}: Loss Curves\n"
                    "Train vs Validation Loss"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                fname = f"{dataset_name}_{ablation}_loss_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating loss plot for {dataset_name} {ablation}: {e}")
                plt.close()
            try:
                plt.figure()
                plt.plot(epochs, ed["alignments"]["train"], label="Train Alignment")
                plt.plot(epochs, ed["alignments"]["val"], label="Validation Alignment")
                plt.title(
                    f"Dataset {dataset_name} - Ablation {ablation}: Alignment Curves\n"
                    "Train vs Validation Alignment"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Alignment")
                plt.legend()
                fname = f"{dataset_name}_{ablation}_alignment_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(
                    f"Error creating alignment plot for {dataset_name} {ablation}: {e}"
                )
                plt.close()
            try:
                plt.figure()
                plt.plot(epochs, ed["mai"], marker="o")
                plt.title(
                    f"Dataset {dataset_name} - Ablation {ablation}: MAI over Epochs\n"
                    "Validation MAI"
                )
                plt.xlabel("Epoch")
                plt.ylabel("MAI")
                fname = f"{dataset_name}_{ablation}_mai.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating MAI plot for {dataset_name} {ablation}: {e}")
                plt.close()
