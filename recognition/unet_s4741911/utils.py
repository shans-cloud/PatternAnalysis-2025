import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Convert a label array to one-hot encoded channels."""
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)

    for c in channels:
        c = int(c)
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

    # Resize image
    first_case = resize(first_case, resize_to, mode="constant", preserve_range=True)

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # Read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[
                :, :, 0
            ]  # Sometimes extra dimension in HipMRI, take first slice

        inImage = resize(inImage, resize_to, mode="constant", preserve_range=True)
        inImage = inImage.astype(dtype)

        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # ~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)  # one-hot encode
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


def show_examples(
    dataset,
    title,
    n=3,
    save_path="sample_visualization.png",
):
    """Display and save n random samples (image + mask pairs) from the dataset."""

    # Randomly choose indices
    indices = random.sample(range(len(dataset)), n)

    fig, axes = plt.subplots(2, n, figsize=(12, 6), gridspec_kw={"wspace": 0.3})
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, idx in enumerate(indices):
        image, seg = dataset[idx]

        # Denormalize image for visualisation
        img_show = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Convert tensors to numpy arrays and plot image
        img_np = img_show.permute(1, 2, 0).squeeze().numpy()  # H x W x C for image

        axes[0, i].imshow(img_np, cmap="gray")
        axes[0, i].set_title(f"Image {idx}", fontweight="bold")
        axes[0, i].axis("off")

        # Debug info for segmentation mask
        seg_np = seg.squeeze(0).numpy()
        unique_vals = np.unique(seg_np)
        seg_pixels = np.sum(seg_np == 1)
        bg_pixels = np.sum(seg_np == 0)

        # Plot segmentation mask
        colours = ["blue", "red"]
        cmap = ListedColormap(colours)
        im = axes[1, i].imshow(seg_np, cmap=cmap, vmin=0, vmax=1)
        axes[1, i].set_title(
            f"Mask {idx} (Seg: {seg_pixels}, BG: {bg_pixels})",
            fontweight="bold",
        )
        axes[1, i].axis("off")

        # Add colorbar for segmentation mask
        if i == 0:
            plt.colorbar(
                im,
                ax=axes[1, i],
                shrink=0.6,
                ticks=[0, 1],
                label="0=BG, 1=Segmentation",
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved sample visualization to: {save_path}")
    return


def show_epoch_predictions(model, dataset, epoch, n=3, device="cuda", indices=None):
    """Show model predictions of validation set after specified epoch."""
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

            # Output of model is sigmoid activated
            pred = model(image.unsqueeze(0).to(device))  # Model expects batch dimension
            pred_mask = (
                pred[0, 0].cpu().numpy()
            )  # 1st batch, 1st channel -> probability map
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

            # Denormalize image for visualisation
            img_show = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Transpose from CHW to HWC for plotting
            img_np = img_show.permute(1, 2, 0).squeeze().cpu().numpy()  # H x W x C

            # Show the original image
            axes[0, i].imshow(img_np, cmap="gray")
            axes[0, i].set_title(f"Image {idx}", fontweight="bold")
            axes[0, i].axis("off")

            # Show the ground truth mask
            axes[1, i].imshow(true_mask.squeeze(0).cpu(), cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title(f"Ground Truth Mask {idx}", fontweight="bold")
            axes[1, i].axis("off")

            # Show the predicted mask
            axes[2, i].imshow(pred_mask_bin, cmap="gray", vmin=0, vmax=1)
            # Pixel accuracy
            accuracy = np.mean(pred_mask_bin == true_mask.squeeze(0).cpu().numpy())
            # Dice coefficient
            true_np = true_mask.squeeze(0).cpu().numpy()
            intersection = np.sum(pred_mask_bin * true_np)
            dice_coeff = (2.0 * intersection + 1e-6) / (
                np.sum(pred_mask_bin) + np.sum(true_np) + 1e-6
            )

            axes[2, i].set_title(
                f"Predicted Mask {idx}\n"
                f"Acc: {accuracy:.2f}\n"
                f"Dice: {dice_coeff:.2f}",
                fontweight="bold",
            )
            axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig(f"epoch_{epoch}_predictions.png")
    plt.close()
    # print(f"Saved epoch {epoch} predictions to: epoch_{epoch}_predictions.png")

    model.train()  # Switch back to train mode
    return


def plot_loss(losses, loss_type="dice"):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, "bo-", linewidth=2, markersize=8)

    title_map = {
        "bce": "Training Loss (BCE)",
        "dice": "Training Loss (Dice)",
        "combined": "Training Loss (Combined BCE + Dice)",
    }

    plt.title(title_map.get(loss_type, "Training Loss"), fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"loss_plot_{loss_type}.png")
    plt.close()
    return


def save_model_checkpoint(state, checkpoint_path="checkpoint.pth.tar"):
    print(f"=> Saving model checkpoint.")
    torch.save(state, checkpoint_path)
    return


def load_model_checkpoint(checkpoint_path="checkpoint.pth.tar", model=None):
    print(f"=> Loading model checkpoint")
    model.load_state_dict(checkpoint_path["state_dict"])
    return
