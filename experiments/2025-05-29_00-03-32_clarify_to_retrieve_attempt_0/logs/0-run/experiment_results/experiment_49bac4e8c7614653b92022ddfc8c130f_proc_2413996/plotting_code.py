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

for name, data in experiment_data.get("user_patience_dropout", {}).items():
    metrics = data.get("metrics", {})
    dr = metrics.get("dropout_rates", [])
    ba = metrics.get("baseline_acc", [])
    ca = metrics.get("clar_acc", [])
    at = metrics.get("avg_turns", [])
    ce = metrics.get("CES", [])
    try:
        plt.figure()
        plt.plot(dr, ba, marker="o", label="Baseline Acc")
        plt.plot(dr, ca, marker="o", label="Clar Acc")
        plt.xlabel("Dropout Rate")
        plt.ylabel("Accuracy")
        plt.title(f"{name} Accuracy vs Dropout Rate\nBaseline vs Clarification")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_accuracy_dropout.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {name}: {e}")
        plt.close()
    try:
        plt.figure()
        plt.plot(dr, at, marker="o")
        plt.xlabel("Dropout Rate")
        plt.ylabel("Average Turns")
        plt.title(f"{name} Avg Turns vs Dropout Rate\nUser Patience Analysis")
        plt.savefig(os.path.join(working_dir, f"{name}_avg_turns_dropout.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating avg turns plot for {name}: {e}")
        plt.close()
    try:
        plt.figure()
        plt.plot(dr, ce, marker="o")
        plt.xlabel("Dropout Rate")
        plt.ylabel("CES")
        plt.title(f"{name} CES vs Dropout Rate\nCost-Effectiveness Score")
        plt.savefig(os.path.join(working_dir, f"{name}_CES_dropout.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CES plot for {name}: {e}")
        plt.close()
    lt = data.get("losses", {}).get("train", [])
    lv = data.get("losses", {}).get("val", [])
    if lt and lv:
        try:
            plt.figure()
            plt.plot(lt, label="Train Loss")
            plt.plot(lv, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{name} Training and Validation Loss\nUser Patience Dropout Experiment"
            )
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{name}_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curves for {name}: {e}")
            plt.close()
