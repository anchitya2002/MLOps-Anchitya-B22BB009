"""
data.py - CIFAR-100 Data Loading and Preprocessing for ViT-S Fine-tuning.

Provides train/val/test data loaders with proper resizing (224x224),
ImageNet normalization, and data augmentation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(img_size=224, is_train=True):
    """Get data transforms for training or evaluation.

    Args:
        img_size: Target image size (default 224 for ViT).
        is_train: Whether to apply training augmentations.

    Returns:
        torchvision.transforms.Compose object.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_cifar100_loaders(data_root="./data", batch_size=64, val_split=0.1,
                         img_size=224, num_workers=4, seed=42):
    """Create CIFAR-100 train, validation, and test data loaders.

    Args:
        data_root: Directory to download/store CIFAR-100.
        batch_size: Batch size for data loaders.
        val_split: Fraction of training data to use for validation.
        img_size: Target image size.
        num_workers: Number of data loader workers.
        seed: Random seed for reproducible train/val split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
    """
    # Training transforms (with augmentation)
    train_transform = get_transforms(img_size=img_size, is_train=True)
    # Evaluation transforms (no augmentation)
    eval_transform = get_transforms(img_size=img_size, is_train=False)

    # Download and load CIFAR-100
    full_train_dataset = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=train_transform
    )
    # Create a separate dataset with eval transforms for validation
    full_val_dataset = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=eval_transform
    )

    # Split training data into train and validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(total_size), [train_size, val_size], generator=generator
    )

    # Use Subset to create train/val splits
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices.indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Get class names
    class_names = full_train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
