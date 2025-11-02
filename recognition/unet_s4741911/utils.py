"""
utils.py
Utility functions for data loading, preprocessing, model checkpointing,
and visualization for UNET-based segmentation of 2D Hip MRI prostate images.
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
import torch


def to_channels(arr: np.ndarray, dtype=np.uint8, num_classes: int = 6) -> np.ndarray:
    """Convert a label array to one-hot encoded channels."""
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)

    for c in range(num_classes):
        res[..., c : c + 1][arr == c] = 1

    return res


# Load medical image functions
def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    early_stop=False,
    resize_to=(256, 128),
):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool ( normalise the image 0.0 -1.0)

    early_stop : Stop loading pre-maturely, leaves arrays mostly empty, for quick
    loading and testing scripts.
    """
    affines = []

    # Get fixed size
    num = len(imageNames)

    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Sometimes extra dimension, take first slice

    # # Resize image
    # first_case = resize(first_case, resize_to, mode="constant", preserve_range=True)

    if categorical:
        first_case = to_channels(first_case, dtype=dtype, num_classes=6)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(
        tqdm(imageNames, desc="Loading images", ncols=100, mininterval=1)
    ):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # Read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            # Sometimes extra dimension in HipMRI, take first slice
            inImage = inImage[:, :, 0]

        inImage = resize(
            inImage,
            resize_to,
            mode="constant",
            preserve_range=True,
        )
        inImage = inImage.astype(dtype)

        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # ~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype, num_classes=6)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def show_epoch_predictions(model, dataset, epoch, n=3, device="cuda", indices=None):
    """
    Show model predictions of validation set after specified epoch.
    Args:
        model: Trained UNET model.
        dataset: Dataset to visualize predictions on.
        epoch: Current epoch number for title.
        n: Number of samples to display.
        device: Device to run model on.
        indices: Specific dataset indices to visualize. If None, random samples are chosen.
    """
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 6))
    fig.suptitle(
        f"Model Predictions after Epoch {epoch}", fontsize=16, fontweight="bold"
    )

    if indices is None:
        indices = random.sample(range(len(dataset)), n)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_mask = dataset[idx]

            # Output of model is logits for multiple classes
            pred = model(image.unsqueeze(0).to(device))  # Model expects batch dimension
            # Predicted class per pixel
            pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()

            # Denormalize image for visualisation
            img_show = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Transpose from CHW to HWC for plotting
            img_np = img_show.permute(1, 2, 0).squeeze().cpu().numpy()  # H x W x C

            # Show the original image
            axes[0, i].imshow(img_np, cmap="gray")
            axes[0, i].set_title(f"Image {idx}", fontweight="bold")
            axes[0, i].axis("off")

            # Show the ground truth mask
            axes[1, i].imshow(true_mask.squeeze(0).cpu(), cmap="gray", vmin=0, vmax=5)
            axes[1, i].set_title(f"Ground Truth Mask {idx}", fontweight="bold")
            axes[1, i].axis("off")

            # Show the predicted mask
            axes[2, i].imshow(pred_mask, cmap="gray", vmin=0, vmax=5)

            true_np = true_mask.squeeze(0).cpu().numpy()
            pred_np = pred_mask

            # Binary masks for prostate class (class 5)
            true_np = (true_np == 5).astype(np.uint8)
            prostate_present = true_np.any()  # Check if prostate is present
            pred_mask = (pred_np == 5).astype(np.uint8)

            intersection = np.logical_and(pred_mask, true_np).sum()
            dice_coeff = (2.0 * intersection) / (pred_mask.sum() + true_np.sum() + 1e-6)

            presence_text = "Present" if prostate_present else "Absent"

            axes[2, i].set_title(
                f"Predicted Mask {idx}\nClass 5 Dice Coeff: {dice_coeff:.3f}\nProstate Presence: {presence_text}",
                fontweight="bold",
            )
            axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig(f"epoch_{epoch}_predictions.png")
    plt.close()

    model.train()  # Switch back to train mode
    return


def plot_loss(train_loss, val_loss, loss_type="dice"):
    """
    Plot training and validation loss curves across epochs.

    Args:
        train_loss (list): List of training loss values per epoch.
        val_loss (list): List of validation loss values per epoch.
        loss_type (str): Type of loss for title and filename.
    """
    plt.figure(figsize=(8, 5))

    # Plot both loss curves
    plt.plot(train_loss, "bo-", label="Training Loss", linewidth=2, markersize=6)
    plt.plot(val_loss, "ro-", label="Validation Loss", linewidth=2, markersize=6)

    title_map = {
        "ce": "Training vs Validation Loss (CE)",
        "dice": "Training vs Validation Loss (Dice)",
        "combined": "Training vs Validation Loss (Combined CE + Dice)",
    }

    plt.title(
        title_map.get(loss_type, "Training vs Validation Loss"),
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"loss_plot_{loss_type}.png")
    plt.close()
    return


def save_model_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save model checkpoint."""
    torch.save(state, filename)
    print(f"=> Saved model checkpoint to {filename}")
    return


def load_model_checkpoint(checkpoint, model):
    """Load model checkpoint."""
    model.load_state_dict(checkpoint["state_dict"])
    print(f"=> Loaded model checkpoint")
    return


def plot_dice_per_class(dice_scores_all_epochs):
    """
    Plot Dice coefficient per class across epochs.
    Args:
        dice_scores_all_epochs (list of lists): Dice scores per class for each epoch.
    """
    dice_scores_all_epochs = np.array(dice_scores_all_epochs)
    epochs = dice_scores_all_epochs.shape[0]
    classes = dice_scores_all_epochs.shape[1]

    plt.figure(figsize=(10, 5))
    for class_idx in range(classes):
        plt.plot(
            range(1, epochs + 1),
            dice_scores_all_epochs[:, class_idx],
            label=f"Class {class_idx}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Dice Coefficient per Class Across Epochs")
    plt.legend()
    plt.savefig("dice_per_class_plot.png")
    plt.close()
    return
