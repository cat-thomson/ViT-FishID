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
import shutil
from pathlib import Path


class SemiSupervisedFishDataset(Dataset):
    """
    Semi-supervised dataset for fish images with both labeled and unlabeled data.
    """
    
    def __init__(
        self,
        labeled_paths: List[str],
        labeled_targets: List[int],
        unlabeled_paths: List[str],
        class_names: List[str],
        transform: Optional[A.Compose] = None,
        image_size: int = 224,
        unlabeled_ratio: float = 1.0
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
        num_unlabeled = int(len(labeled_paths) * unlabeled_ratio)
        self.active_unlabeled_paths = unlabeled_paths[:num_unlabeled]
        
        print(f"Dataset initialized:")
        print(f"  - Labeled samples: {len(labeled_paths)}")
        print(f"  - Unlabeled samples: {len(self.active_unlabeled_paths)}")
        print(f"  - Total samples per epoch: {len(self)}")
    
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


def create_fish_directory_structure(
    input_dir: str,
    output_dir: str,
    labeled_species: List[str] = None,
    copy_files: bool = True
) -> Dict[str, List[str]]:
    """
    Create organized directory structure for fish images.
    
    Args:
        input_dir: Directory containing all fish cutout images
        output_dir: Output directory for organized structure
        labeled_species: List of species you want to label (others go to unlabeled)
        copy_files: Whether to copy files or create symlinks
        
    Returns:
        Dictionary with paths to labeled and unlabeled images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    labeled_dir = output_path / "labeled"
    unlabeled_dir = output_path / "unlabeled"
    
    labeled_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    
    # If labeled_species not provided, use interactive selection
    if labeled_species is None:
        labeled_species = interactive_species_selection(input_dir)
    
    # Create species subdirectories
    for species in labeled_species:
        (labeled_dir / species).mkdir(exist_ok=True)
    
    print(f"Creating fish directory structure...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Labeled species: {labeled_species}")
    
    labeled_paths = []
    unlabeled_paths = []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Process all images in input directory
    for image_path in input_path.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            # Determine if this should be labeled or unlabeled
            species_found = None
            filename_lower = image_path.name.lower()
            
            # Check if filename contains any labeled species name
            for species in labeled_species:
                if species.lower() in filename_lower:
                    species_found = species
                    break
            
            if species_found:
                # Move to labeled directory
                target_path = labeled_dir / species_found / image_path.name
                labeled_paths.append(str(target_path))
            else:
                # Move to unlabeled directory
                target_path = unlabeled_dir / image_path.name
                unlabeled_paths.append(str(target_path))
            
            # Copy or link file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_files:
                shutil.copy2(image_path, target_path)
            else:
                if target_path.exists():
                    target_path.unlink()
                target_path.symlink_to(image_path.absolute())
    
    print(f"\nDirectory structure created:")
    print(f"  - Labeled images: {len(labeled_paths)}")
    print(f"  - Unlabeled images: {len(unlabeled_paths)}")
    
    # Save file lists
    info_file = output_path / "dataset_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Fish Dataset Organization\n")
        f.write(f"========================\n\n")
        f.write(f"Total labeled images: {len(labeled_paths)}\n")
        f.write(f"Total unlabeled images: {len(unlabeled_paths)}\n")
        f.write(f"Labeled species: {', '.join(labeled_species)}\n\n")
        
        for species in labeled_species:
            species_count = sum(1 for p in labeled_paths if f"/{species}/" in p)
            f.write(f"{species}: {species_count} images\n")
    
    return {
        'labeled_paths': labeled_paths,
        'unlabeled_paths': unlabeled_paths,
        'labeled_species': labeled_species
    }


def interactive_species_selection(input_dir: str) -> List[str]:
    """
    Interactive selection of species to label based on filenames.
    """
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Extract potential species names from filenames
    potential_species = set()
    for image_path in input_path.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            # Extract words from filename (assuming species names are in filename)
            filename = image_path.stem.lower()
            words = filename.replace('_', ' ').replace('-', ' ').split()
            potential_species.update(words)
    
    # Filter out common non-species words
    common_words = {
        'fish', 'image', 'img', 'photo', 'pic', 'cutout', 'crop', 'segment',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'jpg', 'png', 'jpeg', 'final', 'processed', 'clean'
    }
    potential_species = [s for s in potential_species if s not in common_words and len(s) > 2]
    potential_species = sorted(potential_species)
    
    print(f"\nFound potential species names in filenames:")
    for i, species in enumerate(potential_species, 1):
        print(f"{i:2d}. {species}")
    
    print(f"\nSelect species to label (others will be unlabeled):")
    print(f"Enter numbers separated by commas (e.g., 1,3,5-7):")
    
    try:
        selection = input("Your selection: ").strip()
        selected_indices = []
        
        for part in selection.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected_indices.extend(range(start-1, end))
            else:
                selected_indices.append(int(part)-1)
        
        selected_species = [potential_species[i] for i in selected_indices 
                          if 0 <= i < len(potential_species)]
        
        print(f"\nSelected species: {selected_species}")
        return selected_species
        
    except (ValueError, IndexError):
        print("Invalid selection. Using first 5 species as default.")
        return potential_species[:5]


def create_semi_supervised_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    unlabeled_ratio: float = 2.0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create semi-supervised data loaders.
    
    Args:
        data_dir: Directory with labeled/ and unlabeled/ subdirectories
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers
        unlabeled_ratio: Ratio of unlabeled to labeled samples
        seed: Random seed
        
    Returns:
        Tuple of (labeled_loader, unlabeled_loader, val_loader, class_names)
    """
    data_path = Path(data_dir)
    labeled_dir = data_path / "labeled"
    unlabeled_dir = data_path / "unlabeled"
    
    # Get class names from labeled directory
    class_names = sorted([d.name for d in labeled_dir.iterdir() if d.is_dir()])
    print(f"Found classes: {class_names}")
    
    # Collect labeled images
    labeled_paths = []
    labeled_targets = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = labeled_dir / class_name
        for image_path in class_dir.glob('*'):
            if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                labeled_paths.append(str(image_path))
                labeled_targets.append(class_idx)
    
    # Collect unlabeled images
    unlabeled_paths = []
    for image_path in unlabeled_dir.glob('*'):
        if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            unlabeled_paths.append(str(image_path))
    
    print(f"Labeled images: {len(labeled_paths)}")
    print(f"Unlabeled images: {len(unlabeled_paths)}")
    
    # Split labeled data into train/val
    np.random.seed(seed)
    indices = np.random.permutation(len(labeled_paths))
    train_size = int(0.8 * len(labeled_paths))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_labeled_paths = [labeled_paths[i] for i in train_indices]
    train_labeled_targets = [labeled_targets[i] for i in train_indices]
    val_labeled_paths = [labeled_paths[i] for i in val_indices]
    val_labeled_targets = [labeled_targets[i] for i in val_indices]
    
    # Create transforms
    strong_transform = create_train_transforms(image_size, strong_augmentation=True)
    weak_transform = create_train_transforms(image_size, strong_augmentation=False)
    val_transform = create_val_transforms(image_size)
    
    # Create datasets
    train_dataset = SemiSupervisedFishDataset(
        train_labeled_paths, train_labeled_targets, unlabeled_paths,
        class_names, strong_transform, image_size, unlabeled_ratio
    )
    
    val_dataset = SemiSupervisedFishDataset(
        val_labeled_paths, val_labeled_targets, [],
        class_names, val_transform, image_size, 0.0
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, class_names


# Import the existing transforms (you'll need to import from data_loader.py)
def create_train_transforms(image_size: int = 224, strong_augmentation: bool = True) -> A.Compose:
    """Create training transforms - same as before."""
    if strong_augmentation:
        return A.Compose([
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def create_val_transforms(image_size: int = 224) -> A.Compose:
    """Create validation transforms - same as before."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


if __name__ == "__main__":
    # Example usage
    input_dir = "/path/to/your/fish/cutouts"  # Your fish cutout directory
    output_dir = "/path/to/organized/fish/dataset"
    
    # Organize fish images
    result = create_fish_directory_structure(
        input_dir=input_dir,
        output_dir=output_dir,
        labeled_species=['salmon', 'trout', 'bass', 'cod'],  # Species you want to label
        copy_files=True
    )
    
    print(f"\nDataset organization complete!")
    print(f"You can now train with: python main_semi_supervised.py --data_dir {output_dir}")
