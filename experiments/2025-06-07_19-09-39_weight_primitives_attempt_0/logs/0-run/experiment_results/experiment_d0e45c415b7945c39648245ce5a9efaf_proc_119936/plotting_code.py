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

try:
    data = experiment_data["synthetic_noise"]["synthetic"]
    metrics_train = data["metrics"]["train"]
    metrics_val = data["metrics"]["val"]
    losses_train = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    predictions = data["predictions"]
    ground_truth = data["ground_truth"]
    noise_levels = data["noise_levels"]
except Exception as e:
    print(f"Error extracting data: {e}")

try:
    plt.figure()
    epochs = metrics_train.shape[1]
    for i, sigma in enumerate(noise_levels):
        plt.plot(range(1, epochs + 1), metrics_train[i], label=f"train σ={sigma}")
        plt.plot(range(1, epochs + 1), metrics_val[i], "--", label=f"val σ={sigma}")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.title("Training and Validation Error Curves - synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves plot: {e}")
    plt.close()

try:
    plt.figure()
    for i, sigma in enumerate(noise_levels):
        plt.plot(range(1, epochs + 1), losses_train[i], label=f"train σ={sigma}")
        plt.plot(range(1, epochs + 1), losses_val[i], "--", label=f"val σ={sigma}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss Curves - synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

for idx, sigma in enumerate(noise_levels):
    try:
        plt.figure()
        sample = 0
        plt.subplot(1, 2, 1)
        plt.plot(ground_truth[idx, sample])
        plt.title("Left: Ground Truth")
        plt.subplot(1, 2, 2)
        plt.plot(predictions[idx, sample])
        plt.title("Right: Generated Samples")
        plt.suptitle(f"Reconstruction for Noise σ={sigma} - synthetic dataset")
        fname = f'synthetic_reconstruction_noise_{str(sigma).replace(".", "p")}.png'
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating reconstruction plot for σ={sigma}: {e}")
        plt.close()
