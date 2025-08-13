import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FishDataset(Dataset):
    """
    Dataset class for fish images.
    """
    
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
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            # Fallback to PIL if cv2 fails
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default preprocessing
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


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
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.7
            ),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.Sharpen(p=0.3),
                A.Emboss(p=0.3),
            ], p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.OneOf([
                A.RandomGamma(p=0.3),
                A.RandomToneCurve(p=0.3),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),
                A.CoarseDropout(
                    max_holes=8, max_height=16, max_width=16,
                    min_holes=1, min_height=8, min_width=8,
                    p=0.3
                ),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    else:
        # Weak augmentation for teacher
        transforms_list = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    
    return A.Compose(transforms_list)


def create_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Create validation transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_fish_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create data loaders for fish classification.
    
    Args:
        data_dir: Directory containing fish images organized by class
        batch_size: Batch size for data loaders
        image_size: Target image size
        num_workers: Number of workers for data loading
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get class names from directory structure
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Collect all image paths and labels
    all_image_paths = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(class_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(class_dir, filename)
                all_image_paths.append(image_path)
                all_labels.append(class_idx)
    
    print(f"Found {len(all_image_paths)} total images")
    
    # Shuffle data
    indices = np.random.permutation(len(all_image_paths))
    all_image_paths = [all_image_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split data
    train_size = int(train_split * len(all_image_paths))
    
    train_paths = all_image_paths[:train_size]
    train_labels = all_labels[:train_size]
    val_paths = all_image_paths[train_size:]
    val_labels = all_labels[train_size:]
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create transforms
    train_transform = create_train_transforms(image_size, strong_augmentation=True)
    val_transform = create_val_transforms(image_size)
    
    # Create datasets
    train_dataset = FishDataset(
        train_paths, train_labels, class_names, train_transform, image_size
    )
    val_dataset = FishDataset(
        val_paths, val_labels, class_names, val_transform, image_size
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
    
    return train_loader, val_loader, class_names


def create_dual_augmentation_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create data loaders with dual augmentation for teacher-student training.
    Each batch contains both strongly and weakly augmented versions.
    
    Args:
        data_dir: Directory containing fish images
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of workers
        train_split: Training split ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # This is a placeholder - you would implement a custom dataset
    # that returns both strong and weak augmentations
    pass
