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


def denormalise_image(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalise a tensor image with ImageNet mean and std."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = tensor * std + mean
    return torch.clamp(denorm_tensor, 0, 1)


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
        img_show = denormalise_image(image)

        # Convert tensors to numpy arrays and plot image
        img_np = img_show.permute(1, 2, 0).numpy()  # H x W x C for image
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Image {idx}", fontweight="bold")
        axes[0, i].axis("off")

        # Debug info for segmentation mask
        seg_np = seg.squeeze(0).numpy()
        unique_vals = np.unique(seg_np)
        seg_pixels = np.sum(seg_np == 1)
        bg_pixels = np.sum(seg_np == 0)

        # Plot segmentation mask
        im = axes[1, i].imshow(seg_np, cmap="RdBu", vmin=0, vmax=1)
        axes[1, i].set_title(
            f"Mask {idx} (Seg: {seg_pixels}, BG: {bg_pixels})",
            fontweight="bold",
        )
        axes[1, i].axis("off")

        # Add colorbar for segmentation mask
        if i == 0:
            colours = ["blue", "red"]
            cmap = ListedColormap(colours)
            im = axes[1, i].imshow(seg_np, cmap=cmap, vmin=0, vmax=1)
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
