import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_data_2D, show_examples
import os
import numpy as np
import torchvision.transforms as transforms


class HipMRIProstateDataset(Dataset):
    """Dataset for Hip MRI Prostate 2D Nifti slices."""

    def __init__(
        self,
        data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
        split="train",
        transform=None,
        subset_size=None,
        normImage=True,
        categorical=True,
        resize_to=(256, 128),
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.normImage = normImage
        self.categorical = categorical
        self.resize_to = resize_to

        # Image and segmentation directories
        img_dir = os.path.join(data_dir, f"keras_slices_{split}")
        seg_dir = os.path.join(data_dir, f"keras_slices_seg_{split}")

        # File names for images and segments
        self.image_files = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        )
        self.seg_files = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir)])

        total_size = len(self.image_files)

        # Handle subset size before loading
        if subset_size is not None and subset_size < total_size:
            self.subset_size = subset_size
            self.image_files = self.image_files[:subset_size]
            self.seg_files = self.seg_files[:subset_size]
            print(
                f"Using subset of {subset_size} samples from {split} split (out of {total_size})."
            )
        else:
            self.subset_size = total_size
            print(f"Using all {total_size} samples from {split} split.")

        # Load resized, normalised arrays
        print(f"Loading {self.subset_size} samples from {split} split...")
        self.images = load_data_2D(
            self.image_files,
            normImage=self.normImage,
            categorical=False,
        )
        self.seg = load_data_2D(
            self.seg_files,
            normImage=False,
            categorical=False,
        )
        print(
            f"Finished loading {len(self.images)} images and {len(self.seg)} segmentations."
        )

    def __len__(self):
        return self.subset_size

    def __getitem__(self, idx):
        image = self.images[idx]
        seg = self.seg[idx]

        # Expand dims to add channel dimension if missing (for grayscale images)
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        if seg.ndim == 2:
            seg = seg[np.newaxis, :, :]

        image = torch.tensor(image, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.uint8)

        if self.transform:
            image = self.transform(image)

        seg_np = seg.numpy()
        binary_seg = np.zeros_like(seg_np, dtype=np.uint8)
        binary_seg[seg_np == 1] = 1  # Prostate class
        binary_seg[seg_np > 1] = 0  # Non-prostate classes

        seg = torch.tensor(binary_seg, dtype=torch.uint8)

        return image, seg


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((256, 128)),  # Standardise size
            transforms.ToTensor(),  # Convert to tensor, scales to [0,1], C x H x W
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet RGB means
                std=[0.229, 0.224, 0.225],  # ImageNet RGB standard deviations
            ),
        ]
    )

    subset_size = 100
    batch_size = 10

    train_dataset = HipMRIProstateDataset(
        data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
        split="train",
        subset_size=subset_size,
        normImage=True,
        categorical=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    show_examples(
        train_dataset,
        title="HipMRI Sample Visualization",
        n=3,
        save_path="train_dataset_examples.png",
    )
