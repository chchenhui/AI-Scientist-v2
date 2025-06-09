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
    experiment_data = {}

for ablation, ds_dict in experiment_data.items():
    for ds_name, sub in ds_dict.items():
        # Print final validation metrics
        try:
            final_acc = sub["metrics"]["val"][-1]
            final_loss = sub["losses"]["val"][-1]
            print(
                f"{ds_name}: final_val_acc={final_acc:.4f}, final_val_loss={final_loss:.4f}"
            )
        except Exception as e:
            print(f"Error printing final metrics for {ds_name}: {e}")

        # Plot validation accuracy and loss curves
        try:
            metrics_val = sub["metrics"].get("val", [])
            losses_val = sub["losses"].get("val", [])
            epochs = list(range(1, len(metrics_val) + 1))
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, metrics_val, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title("Validation Accuracy")
            plt.subplot(1, 2, 2)
            plt.plot(epochs, losses_val, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("Validation Loss")
            plt.suptitle(
                f"{ds_name}: Metrics over Epochs (Text Classification)\n"
                "Left: Validation Accuracy, Right: Validation Loss"
            )
            save_path = os.path.join(working_dir, f"{ds_name}_val_curves.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating plot for {ds_name}: {e}")
            plt.close()
