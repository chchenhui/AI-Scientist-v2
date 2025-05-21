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

drops = list(experiment_data.get("mlp_dropout_rate_ablation", {}).keys())
if drops:
    ds_names = list(experiment_data["mlp_dropout_rate_ablation"][drops[0]].keys())
    for ds in ds_names:
        # Plot loss curves
        try:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            for dk in drops:
                p = dk.split("_", 1)[1]
                train = experiment_data["mlp_dropout_rate_ablation"][dk][ds]["losses"][
                    "train"
                ]
                val = experiment_data["mlp_dropout_rate_ablation"][dk][ds]["losses"][
                    "val"
                ]
                axs[0].plot(range(1, len(train) + 1), train, label=f"drop {p}")
                axs[1].plot(range(1, len(val) + 1), val, label=f"drop {p}")
            fig.suptitle(
                f"Loss Curves on {ds}\nLeft: Training Loss, Right: Validation Loss"
            )
            axs[0].set_title("Training Loss")
            axs[1].set_title("Validation Loss")
            axs[0].set_xlabel("Epoch")
            axs[1].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[1].set_ylabel("Loss")
            axs[0].legend()
            axs[1].legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(working_dir, f"{ds}_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds}: {e}")
            plt.close()
        # Plot alignment curves
        try:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            for dk in drops:
                p = dk.split("_", 1)[1]
                tr = experiment_data["mlp_dropout_rate_ablation"][dk][ds]["alignments"][
                    "train"
                ]
                va = experiment_data["mlp_dropout_rate_ablation"][dk][ds]["alignments"][
                    "val"
                ]
                axs[0].plot(range(1, len(tr) + 1), tr, label=f"drop {p}")
                axs[1].plot(range(1, len(va) + 1), va, label=f"drop {p}")
            fig.suptitle(
                f"Alignment Curves on {ds}\nLeft: Training Alignment, Right: Validation Alignment"
            )
            axs[0].set_title("Training Alignment")
            axs[1].set_title("Validation Alignment")
            axs[0].set_xlabel("Epoch")
            axs[1].set_xlabel("Epoch")
            axs[0].set_ylabel("Alignment")
            axs[1].set_ylabel("Alignment")
            axs[0].legend()
            axs[1].legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(working_dir, f"{ds}_alignment_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating alignment plot for {ds}: {e}")
            plt.close()
        # Plot MAI curves
        try:
            plt.figure(figsize=(6, 4))
            for dk in drops:
                p = dk.split("_", 1)[1]
                mai = experiment_data["mlp_dropout_rate_ablation"][dk][ds]["mai"]
                plt.plot(range(1, len(mai) + 1), mai, label=f"drop {p}")
            plt.title(f"MAI Curves on {ds}\nMAI across Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("MAI")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds}_mai_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating MAI plot for {ds}: {e}")
            plt.close()
