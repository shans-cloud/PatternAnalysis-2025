import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import HipMRIProstateDataset
from modules import UNET
from utils import load_model_checkpoint
from train import DiceLoss
import torchvision.transforms.v2 as v2

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 6
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128
CHECKPOINT_PATH = "checkpoint.pth.tar"

# Class names for visualisation
CLASS_NAMES = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]


# Set random seeds for reproducibility
torch.manual_seed(55)
np.random.seed(55)
random.seed(55)
if torch.cuda.is_available():
    torch.cuda.manual_seed(55)


def evaluate(model, dataloader, device):
    """Evaluate model on test set using Dice coefficient."""
    model.eval()
    dice_fn = DiceLoss(num_classes=NUM_CLASSES)
    dice_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_fn(outputs, masks)
            dice_per_class = dice_fn.get_last_dice_coeff()
            dice_scores.append(dice_per_class)

    dice_scores = np.mean(np.vstack(dice_scores), axis=0)
    mean_dice = np.mean(dice_scores)
    return dice_scores, mean_dice


def visualize_predictions(model, dataset, save_dir="predictions", num_samples=5):
    """Save example predictions as image triplets: input, ground truth, prediction."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    indices = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for idx in indices:
            image, mask = dataset[idx]
            input_tensor = image.unsqueeze(0).to(DEVICE)
            pred = torch.argmax(model(input_tensor), dim=1).squeeze(0).cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(image.squeeze(), cmap="gray")
            axes[0].set_title("Input Image")
            axes[1].imshow(mask.cpu().numpy(), cmap="jet")
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred, cmap="jet")
            axes[2].set_title("Prediction")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"))
            plt.close()


def main():
    # Define transform for test data
    test_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Load test dataset
    test_dataset = HipMRIProstateDataset(
        data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
        split="test",
        transform=test_transform,
        categorical=False,
        normImage=True,
        resize_to=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model and checkpoint
    model = UNET(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    load_model_checkpoint(torch.load(CHECKPOINT_PATH), model)
    print("Loaded model checkpoint successfully.")

    # Evaluate model
    dice_scores, mean_dice = evaluate(model, test_loader, DEVICE)
    print(f"\nDice per class: {dict(zip(CLASS_NAMES, dice_scores.round(3)))}")
    print(f"Mean Dice coefficient: {mean_dice:.4f}")
    print(f"Dice loss (1 - mean Dice): {1 - mean_dice:.4f}")

    # Plot Dice per class
    plt.figure(figsize=(10, 6))
    plt.bar(CLASS_NAMES, dice_scores, color="steelblue")
    plt.ylabel("Dice Coefficient")
    plt.title("Dice Coefficient per Class (Test Set)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("dice_per_class_test.png")
    plt.close()

    # Save qualitative visualisations
    visualize_predictions(
        model, test_dataset, save_dir="visualisation_test", num_samples=6
    )
    print("Saved prediction visualisations to 'visualisation_test/'.")


if __name__ == "__main__":
    main()
