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
        for ds_name, data in ds_dict.items():
            # Loss curves
            try:
                plt.figure()
                plt.plot(data["losses"]["train"], marker="o", label="train")
                plt.plot(data["losses"]["val"], marker="x", label="val")
                plt.title(f"{ds_name} ({ablation}) - Training and Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(
                    os.path.join(working_dir, f"{ds_name}_{ablation}_loss_curve.png")
                )
            except Exception as e:
                print(f"Error creating loss plot for {ds_name} {ablation}: {e}")
            finally:
                plt.close()

            # Memory Retention Ratio
            try:
                plt.figure()
                tr = data["metrics"]["Memory Retention Ratio"]["train"]
                va = data["metrics"]["Memory Retention Ratio"]["val"]
                plt.plot(tr, marker="o", label="train")
                plt.plot(va, marker="x", label="val")
                plt.title(
                    f"{ds_name} ({ablation}) - Memory Retention Ratio\nTrain vs Validation"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Retention Ratio")
                plt.legend()
                plt.savefig(
                    os.path.join(working_dir, f"{ds_name}_{ablation}_mem_retention.png")
                )
            except Exception as e:
                print(
                    f"Error creating memory retention plot for {ds_name} {ablation}: {e}"
                )
            finally:
                plt.close()

            # Entropy-Weighted Memory Efficiency
            try:
                plt.figure()
                tr_e = data["metrics"]["Entropy-Weighted Memory Efficiency"]["train"]
                va_e = data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"]
                plt.plot(tr_e, marker="o", label="train")
                plt.plot(va_e, marker="x", label="val")
                plt.title(
                    f"{ds_name} ({ablation}) - Entropy-Weighted Memory Efficiency\nTrain vs Validation"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Efficiency")
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{ds_name}_{ablation}_eme.png"))
            except Exception as e:
                print(f"Error creating EME plot for {ds_name} {ablation}: {e}")
            finally:
                plt.close()
