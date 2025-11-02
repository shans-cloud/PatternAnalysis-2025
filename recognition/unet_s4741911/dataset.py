"""
dataset.py

Defines dataset class for loading and preprocessing 2D Hip MRI prostate Nifti slices.
It supports optional transformations, normalization, categorical conversion, and resizing.
"""

import torch
from torch.utils.data import Dataset
from utils import load_data_2D
import os
from torchvision.tv_tensors import Image, Mask


class HipMRIProstateDataset(Dataset):
    """
    Dataset for Hip MRI Prostate 2D Nifti slices.

    Handles loading, normalisation, categorical conversion, and resizing.
    """

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
            categorical=self.categorical,
        )

        print(
            f"Finished loading {len(self.images)} images and {len(self.seg)} segmentations."
        )

    def __len__(self):
        return self.subset_size

    def __getitem__(self, idx):
        image = self.images[idx]
        seg = self.seg[idx]

        if self.transform:
            image = Image(torch.tensor(image, dtype=torch.float32))
            seg = Mask(torch.tensor(seg, dtype=torch.int64))
            image, seg = self.transform(image, seg)

        if self.categorical and seg.ndim == 3:
            seg = seg.permute(2, 0, 1)  # Change seg shape to (C, H, W)

        return image, seg
