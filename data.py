"""
Unified Data Loading Module for ViT-FishID.

This module contains:
- FishDataset: Dataset for supervised learning
- SemiSupervisedFishDataset: Dataset for semi-supervised learning  
- Data loading utilities and transformations
- Functions to create train/validation/test data loaders with proper 3-way splitting

Author: GitHub Copilot
Date: 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import random
from typing import List, Tuple, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class FishDataset(Dataset):
    """Dataset class for supervised fish classification."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform: Optional[A.Compose] = None,
        image_size: int = 224
    ):
        """
        Initialize fish dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of labels (class indices)
            class_names: List of class names
            transform: Albumentations transform pipeline
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.image_size = image_size
        
        assert len(image_paths) == len(labels), "Number of images and labels must match"
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset."""
        # Load image
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = self._default_transform(image)
        
        return image, label
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image using cv2 with PIL fallback."""
        image = cv2.imread(image_path)
        
        if image is None:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _default_transform(self, image: np.ndarray) -> torch.Tensor:
        """Default transform if none provided."""
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image


class SemiSupervisedFishDataset(Dataset):
    """Dataset for semi-supervised learning with labeled and unlabeled data."""
    
    def __init__(
        self,
        labeled_paths: List[str],
        labeled_targets: List[int],
        unlabeled_paths: List[str],
        class_names: List[str],
        transform: Optional[A.Compose] = None,
        image_size: int = 224,
        unlabeled_ratio: float = 2.0
    ):
        """
        Initialize semi-supervised dataset.
        
        Args:
            labeled_paths: Paths to labeled images
            labeled_targets: Labels for labeled images
            unlabeled_paths: Paths to unlabeled images
            class_names: List of class names
            transform: Transform pipeline
            image_size: Target image size
            unlabeled_ratio: Ratio of unlabeled to labeled samples per epoch
        """
        self.labeled_paths = labeled_paths
        self.labeled_targets = labeled_targets
        self.unlabeled_paths = unlabeled_paths
        self.class_names = class_names
        self.transform = transform
        self.image_size = image_size
        
        # Calculate how many unlabeled samples to use
        num_unlabeled = min(len(unlabeled_paths), int(len(labeled_paths) * unlabeled_ratio))
        self.active_unlabeled_paths = random.sample(unlabeled_paths, num_unlabeled)
        
        print(f"📊 Dataset initialized:")
        print(f"  - Labeled samples: {len(labeled_paths):,}")
        print(f"  - Unlabeled samples: {len(self.active_unlabeled_paths):,}")
        print(f"  - Total samples per epoch: {len(self):,}")
    
    def __len__(self) -> int:
        return len(self.labeled_paths) + len(self.active_unlabeled_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        """
        Get item from dataset.
        
        Returns:
            Tuple of (image, label_or_pseudo_label, is_labeled)
        """
        if idx < len(self.labeled_paths):
            # Labeled sample
            image_path = self.labeled_paths[idx]
            label = self.labeled_targets[idx]
            is_labeled = True
        else:
            # Unlabeled sample
            unlabeled_idx = idx - len(self.labeled_paths)
            image_path = self.active_unlabeled_paths[unlabeled_idx]
            label = -1  # Placeholder, will be replaced with pseudo-label
            is_labeled = False
        
        # Load image
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = self._default_transform(image)
        
        return image, label, is_labeled
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image using cv2 with PIL fallback."""
        image = cv2.imread(image_path)
        
        if image is None:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _default_transform(self, image: np.ndarray) -> torch.Tensor:
        """Default transform if none provided."""
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image
    
    def resample_unlabeled(self, seed: Optional[int] = None):
        """Resample unlabeled data for the next epoch."""
        if seed is not None:
            random.seed(seed)
        
        num_unlabeled = min(len(self.unlabeled_paths), len(self.active_unlabeled_paths))
        self.active_unlabeled_paths = random.sample(self.unlabeled_paths, num_unlabeled)


def create_train_transforms(image_size: int = 224, strong_augmentation: bool = True) -> A.Compose:
    """
    Create training transforms with strong augmentation for teacher-student training.
    
    Args:
        image_size: Target image size
        strong_augmentation: Whether to apply strong augmentation
        
    Returns:
        Albumentations transform pipeline
    """
    if strong_augmentation:
        # Strong augmentation for student
        transforms_list = [
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CLAHE(clip_limit=2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    else:
        # Weak augmentation for teacher
        transforms_list = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    
    return A.Compose(transforms_list)


def create_val_transforms(image_size: int = 224) -> A.Compose:
    """Create validation transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def scan_dataset_directory(data_dir: str) -> Dict[str, Any]:
    """
    Scan dataset directory and return paths and class information.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Dictionary with dataset information
    """
    data_dir = Path(data_dir)
    
    # Check for semi-supervised structure (labeled/ and unlabeled/)
    labeled_dir = data_dir / "labeled"
    unlabeled_dir = data_dir / "unlabeled"
    
    if labeled_dir.exists() and unlabeled_dir.exists():
        # Semi-supervised structure
        labeled_paths, labeled_targets, class_names = _scan_labeled_directory(labeled_dir)
        unlabeled_paths = _scan_unlabeled_directory(unlabeled_dir)
        
        return {
            'type': 'semi_supervised',
            'labeled_paths': labeled_paths,
            'labeled_targets': labeled_targets,
            'unlabeled_paths': unlabeled_paths,
            'class_names': class_names,
            'num_classes': len(class_names)
        }
    else:
        # Supervised structure (class folders directly in data_dir)
        labeled_paths, labeled_targets, class_names = _scan_labeled_directory(data_dir)
        
        return {
            'type': 'supervised',
            'labeled_paths': labeled_paths,
            'labeled_targets': labeled_targets,
            'class_names': class_names,
            'num_classes': len(class_names)
        }


def _scan_labeled_directory(labeled_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    """Scan labeled directory and return paths, targets, and class names."""
    labeled_paths = []
    labeled_targets = []
    class_names = []
    
    # Get class directories
    class_dirs = [d for d in labeled_dir.iterdir() if d.is_dir()]
    class_dirs.sort()
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        
        # Get all image files in this class
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        class_images = [
            str(img_path) for img_path in class_dir.iterdir()
            if img_path.suffix.lower() in image_extensions
        ]
        
        labeled_paths.extend(class_images)
        labeled_targets.extend([class_idx] * len(class_images))
    
    return labeled_paths, labeled_targets, class_names


def _scan_unlabeled_directory(unlabeled_dir: Path) -> List[str]:
    """Scan unlabeled directory and return image paths."""
    unlabeled_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for img_path in unlabeled_dir.rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            unlabeled_paths.append(str(img_path))
    
    return unlabeled_paths


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create supervised train, validation, and test data loaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for test
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Scan dataset
    dataset_info = scan_dataset_directory(data_dir)
    
    if dataset_info['type'] == 'semi_supervised':
        # Use only labeled data for supervised training
        labeled_paths = dataset_info['labeled_paths']
        labeled_targets = dataset_info['labeled_targets']
    else:
        labeled_paths = dataset_info['labeled_paths']
        labeled_targets = dataset_info['labeled_targets']
    
    class_names = dataset_info['class_names']
    
    # First split: separate test set
    from sklearn.model_selection import train_test_split
    
    temp_paths, test_paths, temp_targets, test_targets = train_test_split(
        labeled_paths, labeled_targets,
        test_size=test_split,
        stratify=labeled_targets,
        random_state=seed
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust val_split to account for already removed test data
    adjusted_val_split = val_split / (1 - test_split)
    
    train_paths, val_paths, train_targets, val_targets = train_test_split(
        temp_paths, temp_targets,
        test_size=adjusted_val_split,
        stratify=temp_targets,
        random_state=seed + 1  # Different seed for second split
    )
    
    # Create transforms
    train_transform = create_train_transforms(image_size, strong_augmentation=True)
    val_transform = create_val_transforms(image_size)
    test_transform = create_val_transforms(image_size)  # Same as val transforms
    
    # Create datasets
    train_dataset = FishDataset(
        image_paths=train_paths,
        labels=train_targets,
        class_names=class_names,
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = FishDataset(
        image_paths=val_paths,
        labels=val_targets,
        class_names=class_names,
        transform=val_transform,
        image_size=image_size
    )
    
    test_dataset = FishDataset(
        image_paths=test_paths,
        labels=test_targets,
        class_names=class_names,
        transform=test_transform,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"📊 Supervised data loaders created:")
    print(f"  - Train samples: {len(train_dataset):,}")
    print(f"  - Val samples: {len(val_dataset):,}")
    print(f"  - Test samples: {len(test_dataset):,}")
    print(f"  - Classes: {len(class_names)}")
    print(f"  - Split ratios: Train={1-val_split-test_split:.1%}, Val={val_split:.1%}, Test={test_split:.1%}")
    
    return train_loader, val_loader, test_loader, class_names


def create_semi_supervised_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.2,
    unlabeled_ratio: float = 2.0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int, int]:
    """
    Create semi-supervised train, validation, and test data loaders.
    
    Args:
        data_dir: Path to dataset directory with labeled/ and unlabeled/ subdirs
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers for data loading
        val_split: Fraction of labeled data to use for validation
        test_split: Fraction of labeled data to use for test
        unlabeled_ratio: Ratio of unlabeled to labeled samples per epoch
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, labeled_count, unlabeled_count)
    """
    # Scan dataset
    dataset_info = scan_dataset_directory(data_dir)
    
    if dataset_info['type'] != 'semi_supervised':
        raise ValueError(f"Data directory must have 'labeled/' and 'unlabeled/' subdirectories")
    
    labeled_paths = dataset_info['labeled_paths']
    labeled_targets = dataset_info['labeled_targets']
    unlabeled_paths = dataset_info['unlabeled_paths']
    class_names = dataset_info['class_names']
    
    # First split: separate test set from labeled data
    from sklearn.model_selection import train_test_split
    
    temp_paths, test_paths, temp_targets, test_targets = train_test_split(
        labeled_paths, labeled_targets,
        test_size=test_split,
        stratify=labeled_targets,
        random_state=seed
    )
    
    # Second split: separate train and validation from remaining labeled data
    # Adjust val_split to account for already removed test data
    adjusted_val_split = val_split / (1 - test_split)
    
    train_paths, val_paths, train_targets, val_targets = train_test_split(
        temp_paths, temp_targets,
        test_size=adjusted_val_split,
        stratify=temp_targets,
        random_state=seed + 1  # Different seed for second split
    )
    
    # Create transforms
    train_transform = create_train_transforms(image_size, strong_augmentation=True)
    val_transform = create_val_transforms(image_size)
    test_transform = create_val_transforms(image_size)  # Same as val transforms
    
    # Create semi-supervised training dataset
    train_dataset = SemiSupervisedFishDataset(
        labeled_paths=train_paths,
        labeled_targets=train_targets,
        unlabeled_paths=unlabeled_paths,
        class_names=class_names,
        transform=train_transform,
        image_size=image_size,
        unlabeled_ratio=unlabeled_ratio
    )
    
    # Create validation dataset (labeled only)
    val_dataset = FishDataset(
        image_paths=val_paths,
        labels=val_targets,
        class_names=class_names,
        transform=val_transform,
        image_size=image_size
    )
    
    # Create test dataset (labeled only)
    test_dataset = FishDataset(
        image_paths=test_paths,
        labels=test_targets,
        class_names=class_names,
        transform=test_transform,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    labeled_count = len(train_paths)
    unlabeled_count = len(train_dataset.active_unlabeled_paths)
    
    print(f"📊 Semi-supervised data loaders created:")
    print(f"  - Train labeled: {labeled_count:,}")
    print(f"  - Train unlabeled: {unlabeled_count:,}")
    print(f"  - Val samples: {len(val_dataset):,}")
    print(f"  - Test samples: {len(test_dataset):,}")
    print(f"  - Classes: {len(class_names)}")
    print(f"  - Split ratios: Train={1-val_split-test_split:.1%}, Val={val_split:.1%}, Test={test_split:.1%}")
    
    return train_loader, val_loader, test_loader, class_names, labeled_count, unlabeled_count
