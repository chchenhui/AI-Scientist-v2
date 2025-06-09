import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_data = {}

exp = exp_data.get("lstm_dropout", {}).get("synthetic", {})
dropouts = exp.get("dropout_rates", [])
loss_tr = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
acc_tr = exp.get("metrics", {}).get("train", [])
acc_val = exp.get("metrics", {}).get("val", [])

# 1: Loss curves
try:
    plt.figure()
    for d, lt, lv in zip(dropouts, loss_tr, loss_val):
        plt.plot(lt, linestyle="-", label=f"train dr={d}")
        plt.plot(lv, linestyle="--", label=f"val dr={d}")
    plt.title("Loss Curves (synthetic)\nTrain: solid; Val: dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves_lstm_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# 2: Accuracy curves
try:
    plt.figure()
    for d, at, av in zip(dropouts, acc_tr, acc_val):
        plt.plot(at, linestyle="-", label=f"train dr={d}")
        plt.plot(av, linestyle="--", label=f"val dr={d}")
    plt.title("Accuracy Curves (synthetic)\nTrain: solid; Val: dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curves_lstm_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# 3: Final validation accuracy bar chart
try:
    final_acc = [av[-1] for av in acc_val]
    plt.figure()
    plt.bar([str(d) for d in dropouts], final_acc)
    plt.title(
        "Final Validation Accuracy by Dropout Rate (synthetic)\nBar height = val accuracy"
    )
    plt.xlabel("LSTM Dropout Rate")
    plt.ylabel("Validation Accuracy")
    plt.savefig(
        os.path.join(working_dir, "synthetic_final_val_accuracy_lstm_dropout.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
