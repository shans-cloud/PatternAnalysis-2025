"""
train.py
Training script for UNET-based segmentation of 2D Hip MRI prostate images.
Includes data loading, model training loop, validation, checkpointing, and visualization.

Key Features:
- Uses Focal + Dice Loss for handling class imbalance.
- train() function encapsulates the training process.
- main() function sets up datasets, model, optimizer, and starts training.
"""

import random
import os
import torch
from torch.utils.data import DataLoader
from dataset import HipMRIProstateDataset
import torchvision.transforms.v2 as v2
from utils import (
    plot_loss,
    show_epoch_predictions,
    save_model_checkpoint,
    load_model_checkpoint,
    plot_dice_per_class,
)
import torch.optim as optim
from modules import UNET
import numpy as np


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBSET_SIZE_TRAIN = None  # Use None to load full dataset
SUBSET_SIZE_VAL = None  # Use None to load full dataset
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 64
NUM_EPOCHS = 20  # 100
NUM_WORKERS = 1
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128
NUM_CLASSES = 6

# Set random seeds for reproducibility
torch.manual_seed(55)
np.random.seed(55)
random.seed(55)
if torch.cuda.is_available():
    torch.cuda.manual_seed(55)


class DiceLoss(torch.nn.Module):
    """
    Dice loss implementation for multi-class segmentation.
    Computes Dice coefficient per class and averages with optional class weights.

    Args:
        smooth (float): Smoothing factor to avoid division by zero.
        num_classes (int): Number of segmentation classes.
        class_weights (torch.Tensor, optional): Weights for each class to handle imbalance.

    Returns:
        torch.Tensor: scalarDice loss value.
    """

    def __init__(self, smooth=1e-5, num_classes=6, class_weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.last_dice_coeff = None
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        # Apply softmax for probabilities
        inputs = torch.softmax(inputs, dim=1)

        # One-hot encode targets
        targets_one_hot = (
            torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Flatten for spatial dimensions
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_one_hot = targets_one_hot.view(
            targets_one_hot.size(0), self.num_classes, -1
        )

        # Compute intersection and union for Dice coefficient
        intersection = (inputs * targets_one_hot).sum(-1)
        total = inputs.sum(-1) + targets_one_hot.sum(-1)
        dice_per_class = (2.0 * intersection + self.smooth) / (total + self.smooth)

        # Mask out classes not present in ground truth
        mask = (targets_one_hot.sum(-1) > 0).float()
        dice_per_class = (dice_per_class * mask).sum(0) / mask.sum(0).clamp(min=1.0)

        self.last_dice_coeff = dice_per_class.detach().cpu().numpy()

        # Apply class weights if provided
        if self.class_weights is not None:
            dice_loss = (1 - dice_per_class) * self.class_weights
            dice_loss = dice_loss.sum() / self.class_weights.sum()
        else:
            dice_loss = 1 - dice_per_class.mean()

        return dice_loss

    def get_last_dice_coeff(self):
        return self.last_dice_coeff


class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for multi-class segmentation.
    Focuses training on hard-to-classify examples.

    Args:
        weight (torch.Tensor, optional): Weights for each class.
        gamma (float): Focusing parameter.

    Returns:
        torch.Tensor: scalar Focal loss value.
    """

    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, inputs, targets):
        # Compute per-pixel cross entropy loss
        ce_loss = self.ce(inputs, targets)  # [B, H, W]
        pt = torch.exp(-ce_loss)  # model confidence for the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class FocalDiceLoss(torch.nn.Module):
    """
    Combined Focal and Dice Loss for multi-class segmentation.

    Args:
        weight (torch.Tensor, optional): Weights for each class.
        dice_factor (float): Weighting factor for Dice loss component.
        focal_factor (float): Weighting factor for Focal loss component.
        num_classes (int): Number of segmentation classes.

    Returns:
        torch.Tensor: scalar combined loss value.
    """

    def __init__(self, weight=None, dice_factor=3.0, focal_factor=1.0, num_classes=6):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(weight=weight)
        self.dice = DiceLoss(num_classes=num_classes, class_weights=weight)
        self.dice_factor = dice_factor
        self.focal_factor = focal_factor
        self.last_dice_coeff = None

    def forward(self, inputs, targets):
        # Compute focal and dice losses
        focal = self.focal(inputs, targets)
        dice = self.dice(inputs, targets)
        self.last_dice_coeff = self.dice.get_last_dice_coeff()
        # Returned weighted sum of losses
        return self.focal_factor * focal + self.dice_factor * dice

    def get_last_dice_coeff(self):
        return self.last_dice_coeff


def train(
    model,
    train_loader,
    validation_loader,
    visualize_every=20,
    criterion=None,
    optimizer=None,
):
    """
    Main training loop for UNET model.
    """
    train_losses = []
    val_losses = []
    dice_scores_all_epochs = []

    scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

    # Fixed validation samples for visualisation across epochs
    val_dataset = validation_loader.dataset
    _vis_n = 3  # number of samples to visualise
    _vis_indices = random.sample(range(len(val_dataset)), _vis_n)

    print(
        "Starting training with Batch Norm, ReLU, Dropout, using Focal and Dice Loss..."
    )

    # Load from checkpoint if available
    checkpoint_path = "checkpoint.pth.tar"
    if os.path.exists(checkpoint_path):
        try:
            load_model_checkpoint(torch.load(checkpoint_path), model)
        except Exception as e:
            print(f"Could not load checkpoint ({e}). Starting from scratch.")
    else:
        print("No checkpoint found. Starting from scratch.")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(enabled=(scaler is not None), device_type=DEVICE):
                outputs = model(images)  # [B, C, H, W]
                loss = criterion(outputs, masks)

            # Backward pass and optimization
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            dice_scores = []

            for idx, (val_images, val_masks) in enumerate(validation_loader):
                val_images, val_masks = val_images.to(DEVICE), val_masks.to(DEVICE)

                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_masks)
                val_loss += loss.item()

                # Collect per-class Dice scores
                if hasattr(criterion, "get_last_dice_coeff"):
                    dice_per_class = criterion.get_last_dice_coeff()
                    if dice_per_class is not None:
                        dice_scores.append(dice_per_class)

            # Compute average Dice per class across validation batches
            if dice_scores:
                dice_scores = np.mean(np.vstack(dice_scores), axis=0)
                dice_scores_all_epochs.append(dice_scores)
                print(f"Dice per class at epoch: {dice_scores}")

            avg_val_loss = val_loss / len(validation_loader)
            val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        # Save checkpoint if best validation loss improves
        if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_model_checkpoint(checkpoint)
            print(
                f"Saved new best model checkpoint at epoch {epoch + 1} (val loss: {avg_val_loss:.4f})"
            )

        # Visualize predictions
        if (epoch + 1) % visualize_every == 0:
            show_epoch_predictions(model, val_dataset, epoch + 1, indices=_vis_indices)

    print("Training complete with U-NET architecture.")

    plot_dice_per_class(dice_scores_all_epochs)

    return train_losses, val_losses


def main():

    train_transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(-10, 10)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    train_dataset = HipMRIProstateDataset(
        data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
        split="train",
        transform=train_transform,
        subset_size=SUBSET_SIZE_TRAIN,
        categorical=False,  # (256, 128) masks
        normImage=True,
        resize_to=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    # Shape checking
    sample_image, sample_mask = train_dataset[0]
    print(f"Sample image shape: {sample_image.shape}")  # ([1, 256, 128])
    print(f"Sample mask shape: {sample_mask.shape}")  # ([256, 128])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    validate_dataset = HipMRIProstateDataset(
        data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
        split="validate",
        transform=val_transform,
        subset_size=SUBSET_SIZE_VAL,
        normImage=True,
        categorical=False,
        resize_to=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Compute class weights for CrossEntropyLoss
    class_weights = torch.tensor([1, 1, 1, 3, 9, 10], dtype=torch.float).to(DEVICE)

    print(f"Class weights: {class_weights}")

    model = UNET(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    criterion = FocalDiceLoss(
        weight=class_weights, dice_factor=3.0, focal_factor=1.0, num_classes=NUM_CLASSES
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = train(
        model,
        train_loader,
        validate_loader,
        visualize_every=10,
        criterion=criterion,
        optimizer=optimizer,
    )
    plot_loss(train_losses, val_losses, loss_type="combined")


if __name__ == "__main__":
    main()
