import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    acts = list(exp.keys())
    synth = {act: exp[act]["synthetic"] for act in acts}
except Exception as e:
    print(f"Error loading experiment data: {e}")
    synth = {}

# Compute and print test accuracies
test_accs = {}
for act, data in synth.items():
    preds = data["predictions"]
    gt = data["ground_truth"]
    acc = np.mean(preds == gt)
    test_accs[act] = acc
    print(f"{act} test accuracy: {acc:.4f}")

# 1. AI loss curves
try:
    plt.figure()
    for act, data in synth.items():
        plt.plot(data["ai_losses"]["train"], label=f"{act} train")
        plt.plot(data["ai_losses"]["val"], linestyle="--", label=f"{act} val")
    plt.title("AI Loss Curves on Synthetic Dataset\nSolid: Train, Dashed: Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‐Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_ai_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# 2. AI accuracy curves
try:
    plt.figure()
    for act, data in synth.items():
        plt.plot(data["ai_metrics"]["train"], label=f"{act} train")
        plt.plot(data["ai_metrics"]["val"], linestyle="--", label=f"{act} val")
    plt.title("AI Accuracy Curves on Synthetic Dataset\nSolid: Train, Dashed: Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_ai_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# 3. User loss curves
try:
    plt.figure()
    for act, data in synth.items():
        plt.plot(data["losses"]["train"], label=f"{act} train")
        plt.plot(data["losses"]["val"], linestyle="--", label=f"{act} val")
    plt.title("User Loss Curves on Synthetic Dataset\nSolid: Train, Dashed: Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‐Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_user_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# 4. User accuracy curves
try:
    plt.figure()
    for act, data in synth.items():
        plt.plot(data["metrics"]["train"], label=f"{act} train")
        plt.plot(data["metrics"]["val"], linestyle="--", label=f"{act} val")
    plt.title("User Accuracy Curves on Synthetic Dataset\nSolid: Train, Dashed: Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_user_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# 5. Test accuracy bar chart
try:
    plt.figure()
    acts_list = list(test_accs.keys())
    acc_list = [test_accs[a] for a in acts_list]
    plt.bar(acts_list, acc_list)
    plt.title("Test Accuracy by Activation on Synthetic Dataset")
    plt.xlabel("Activation")
    plt.ylabel("Test Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()
