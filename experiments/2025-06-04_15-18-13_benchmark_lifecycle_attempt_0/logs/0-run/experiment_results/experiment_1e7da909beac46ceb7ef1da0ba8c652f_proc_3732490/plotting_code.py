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

# Print final metrics for each dataset
for ds, data in experiment_data.items():
    final_val = data["losses"]["val"][-1]
    final_disc = data["disc_score"][-1]
    print(
        f"{ds}: final validation loss = {final_val:.4f}, discrimination score = {final_disc:.4f}"
    )

# Plot 1: training and validation loss curves comparison
try:
    plt.figure()
    for ds, data in experiment_data.items():
        epochs = np.arange(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label=f"{ds} train")
        plt.plot(epochs, data["losses"]["val"], linestyle="--", label=f"{ds} val")
    plt.title("Training and Validation Loss Curves Across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss comparison plot: {e}")
    plt.close()

# Plot 2: discrimination score curves
try:
    plt.figure()
    for ds, data in experiment_data.items():
        epochs = np.arange(1, len(data["disc_score"]) + 1)
        plt.plot(epochs, data["disc_score"], label=ds)
    plt.title("Benchmark Discrimination Score Curves Across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Discrimination Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "disc_score_curves_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating discrimination score plot: {e}")
    plt.close()

# Plot 3: final validation loss bar chart
try:
    plt.figure()
    ds_list = list(experiment_data.keys())
    vals = [experiment_data[ds]["losses"]["val"][-1] for ds in ds_list]
    x = np.arange(len(ds_list))
    plt.bar(x, vals)
    plt.xticks(x, ds_list)
    plt.title("Final Validation Loss per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "final_validation_loss_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final validation loss bar plot: {e}")
    plt.close()

# Plot 4: final discrimination score bar chart
try:
    plt.figure()
    scores = [experiment_data[ds]["disc_score"][-1] for ds in ds_list]
    x = np.arange(len(ds_list))
    plt.bar(x, scores)
    plt.xticks(x, ds_list)
    plt.title("Final Discrimination Score per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Discrimination Score")
    plt.savefig(os.path.join(working_dir, "final_discrimination_score_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final discrimination score bar plot: {e}")
    plt.close()
