#!/usr/bin/env python3
"""
Semi-supervised training script for ViT-Base with EMA Teacher-Student framework.

This script implements semi-supervised learning using both labeled and unlabeled
fish images. The EMA teacher-student framework leverages unlabeled data through
consistency regularization.

Usage:
    # First, organize your fish cutouts:
    python organize_fish_data.py --input_dir /path/to/fish/cutouts --output_dir /path/to/organized/dataset
    
    # Then train:
    python main_semi_supervised.py --data_dir /path/to/organized/dataset --epochs 100

Author: GitHub Copilot
Date: 2025
"""

import argparse
import os
import sys
import torch
import wandb
from datetime import datetime
from typing import Tuple

# Import custom modules
from vit_model import ViTForFishClassification
from semi_supervised_trainer import SemiSupervisedEMATrainer
from semi_supervised_data import create_semi_supervised_dataloaders
from utils import set_seed, get_device, count_parameters


def parse_arguments():
    """Parse command line arguments for semi-supervised training."""
    parser = argparse.ArgumentParser(description='Semi-Supervised ViT Training with EMA Teacher-Student')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to organized dataset with labeled/ and unlabeled/ directories')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--unlabeled_ratio', type=float, default=2.0,
                        help='Ratio of unlabeled to labeled samples per epoch (default: 2.0)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='ViT model architecture (default: vit_base_patch16_224)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights (default: True)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate for classification head (default: 0.1)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs (default: 10)')
    parser.add_argument('--ramp_up_epochs', type=int, default=20,
                        help='Epochs to ramp up consistency weight (default: 20)')
    
    # Semi-supervised EMA arguments
    parser.add_argument('--ema_momentum', type=float, default=0.999,
                        help='EMA momentum for teacher updates (default: 0.999)')
    parser.add_argument('--consistency_loss', type=str, default='mse', choices=['mse', 'kl'],
                        help='Type of consistency loss (default: mse)')
    parser.add_argument('--consistency_weight', type=float, default=1.0,
                        help='Weight for consistency loss (default: 1.0)')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.95,
                        help='Confidence threshold for pseudo-labels (default: 0.95)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for consistency loss (default: 4.0)')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu). If None, auto-detect')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./semi_supervised_checkpoints',
                        help='Directory to save checkpoints (default: ./semi_supervised_checkpoints)')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='vit-fish-semi-supervised',
                        help='Wandb project name (default: vit-fish-semi-supervised)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def setup_wandb(args, class_names, labeled_count, unlabeled_count):
    """Setup Weights & Biases logging for semi-supervised training."""
    if not args.use_wandb:
        return
    
    # Generate run name if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.wandb_run_name = f"semi_ema_vit_{args.model_name}_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            'model_name': args.model_name,
            'image_size': args.image_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'ema_momentum': args.ema_momentum,
            'consistency_loss': args.consistency_loss,
            'consistency_weight': args.consistency_weight,
            'pseudo_label_threshold': args.pseudo_label_threshold,
            'temperature': args.temperature,
            'unlabeled_ratio': args.unlabeled_ratio,
            'num_classes': len(class_names),
            'class_names': class_names,
            'labeled_samples': labeled_count,
            'unlabeled_samples': unlabeled_count,
            'pretrained': args.pretrained,
            'dropout_rate': args.dropout_rate,
            'warmup_epochs': args.warmup_epochs,
            'ramp_up_epochs': args.ramp_up_epochs,
            'training_type': 'semi_supervised'
        }
    )
    
    print(f"Wandb initialized: {args.wandb_project}/{args.wandb_run_name}")


def validate_data_directory(data_dir: str) -> Tuple[bool, str]:
    """
    Validate that the data directory has the correct structure.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(data_dir):
        return False, f"Data directory does not exist: {data_dir}"
    
    labeled_dir = os.path.join(data_dir, 'labeled')
    unlabeled_dir = os.path.join(data_dir, 'unlabeled')
    
    if not os.path.exists(labeled_dir):
        return False, f"Labeled directory not found: {labeled_dir}"
    
    if not os.path.exists(unlabeled_dir):
        return False, f"Unlabeled directory not found: {unlabeled_dir}"
    
    # Check if labeled directory has subdirectories (species)
    species_dirs = [d for d in os.listdir(labeled_dir) 
                   if os.path.isdir(os.path.join(labeled_dir, d))]
    
    if len(species_dirs) == 0:
        return False, f"No species directories found in {labeled_dir}"
    
    # Check if unlabeled directory has images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    unlabeled_images = [f for f in os.listdir(unlabeled_dir) 
                       if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if len(unlabeled_images) == 0:
        return False, f"No images found in {unlabeled_dir}"
    
    return True, ""


def count_dataset_samples(data_dir: str) -> Tuple[int, int]:
    """Count labeled and unlabeled samples."""
    labeled_dir = os.path.join(data_dir, 'labeled')
    unlabeled_dir = os.path.join(data_dir, 'unlabeled')
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Count labeled samples
    labeled_count = 0
    for species_dir in os.listdir(labeled_dir):
        species_path = os.path.join(labeled_dir, species_dir)
        if os.path.isdir(species_path):
            species_images = [f for f in os.listdir(species_path) 
                            if any(f.lower().endswith(ext) for ext in image_extensions)]
            labeled_count += len(species_images)
    
    # Count unlabeled samples
    unlabeled_images = [f for f in os.listdir(unlabeled_dir) 
                       if any(f.lower().endswith(ext) for ext in image_extensions)]
    unlabeled_count = len(unlabeled_images)
    
    return labeled_count, unlabeled_count


def main():
    """Main semi-supervised training function."""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    if args.device is None:
        device = get_device()
    else:
        device = args.device
    
    print(f"🐟 Starting ViT-Base Semi-Supervised EMA Training")
    print(f"Arguments: {vars(args)}")
    
    # Validate data directory structure
    is_valid, error_msg = validate_data_directory(args.data_dir)
    if not is_valid:
        print(f"❌ Invalid data directory structure: {error_msg}")
        print(f"\nExpected structure:")
        print(f"  {args.data_dir}/")
        print(f"  ├── labeled/")
        print(f"  │   ├── species_1/")
        print(f"  │   │   ├── img1.jpg")
        print(f"  │   │   └── img2.jpg")
        print(f"  │   └── species_2/")
        print(f"  │       └── ...")
        print(f"  └── unlabeled/")
        print(f"      ├── fish_img1.jpg")
        print(f"      ├── fish_img2.jpg")
        print(f"      └── ...")
        print(f"\nUse the organize_fish_data.py script to create this structure.")
        sys.exit(1)
    
    # Count samples
    labeled_count, unlabeled_count = count_dataset_samples(args.data_dir)
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Labeled samples: {labeled_count:,}")
    print(f"  - Unlabeled samples: {unlabeled_count:,}")
    print(f"  - Unlabeled ratio: {args.unlabeled_ratio}x")
    print(f"  - Effective unlabeled per epoch: {int(labeled_count * args.unlabeled_ratio):,}")
    
    # Create data loaders
    print(f"\n🔄 Creating semi-supervised data loaders...")
    train_loader, val_loader, class_names = create_semi_supervised_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        unlabeled_ratio=args.unlabeled_ratio,
        seed=args.seed
    )
    
    num_classes = len(class_names)
    print(f"\n🏷️ Classes ({num_classes}): {class_names}")
    
    # Create model
    print(f"\n🧠 Creating ViT model: {args.model_name}")
    student_model = ViTForFishClassification(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate
    )
    
    # Print model info
    total_params = count_parameters(student_model)
    print(f"Model parameters: {total_params:,}")
    
    # Setup wandb
    setup_wandb(args, class_names, labeled_count, unlabeled_count)
    
    # Create trainer
    print(f"\n🎯 Initializing Semi-Supervised EMA Trainer...")
    trainer = SemiSupervisedEMATrainer(
        student_model=student_model,
        num_classes=num_classes,
        device=device,
        ema_momentum=args.ema_momentum,
        consistency_loss_type=args.consistency_loss,
        consistency_weight=args.consistency_weight,
        pseudo_label_threshold=args.pseudo_label_threshold,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        ramp_up_epochs=args.ramp_up_epochs,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        # Implementation would load checkpoint here
        pass
    
    # Start training
    print(f"\n🚀 Starting semi-supervised training for {args.epochs} epochs...")
    print(f"💡 Key Features:")
    print(f"  - EMA momentum: {args.ema_momentum}")
    print(f"  - Consistency weight: {args.consistency_weight} (ramped up over {args.ramp_up_epochs} epochs)")
    print(f"  - Pseudo-label threshold: {args.pseudo_label_threshold}")
    print(f"  - Temperature: {args.temperature}")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency
    )
    
    print(f"\n✅ Semi-supervised training completed!")
    print(f"📁 Checkpoints saved to: {args.save_dir}")
    
    # Finish wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
