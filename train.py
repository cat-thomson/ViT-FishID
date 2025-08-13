#!/usr/bin/env python3
"""
Unified Training Script for ViT-FishID.

This script supports both supervised and semi-supervised training modes:
- Supervised: Uses only labeled data with EMA teacher-student framework
- Semi-supervised: Uses both labeled and unlabeled data

Usage:
    # Supervised training
    python train.py --data_dir /path/to/fish/dataset --mode supervised
    
    # Semi-supervised training
    python train.py --data_dir /path/to/organized/dataset --mode semi_supervised

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

# Import unified modules
from model import create_model_and_teacher
from trainer import EMATrainer, SemiSupervisedTrainer, train_model
from data import create_dataloaders, create_semi_supervised_dataloaders
from utils import set_seed, get_device, count_parameters


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ViT-FishID Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--mode', type=str, choices=['supervised', 'semi_supervised'], 
                        default='semi_supervised',
                        help='Training mode: supervised or semi_supervised')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test split ratio (default: 0.2)')
    
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
    
    # EMA arguments
    parser.add_argument('--ema_momentum', type=float, default=0.999,
                        help='EMA momentum for teacher (default: 0.999)')
    parser.add_argument('--consistency_weight', type=float, default=2.0,
                        help='Weight for consistency loss (default: 2.0)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for consistency loss (default: 4.0)')
    
    # Semi-supervised arguments
    parser.add_argument('--unlabeled_ratio', type=float, default=2.0,
                        help='Ratio of unlabeled to labeled samples per epoch (default: 2.0)')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.7,
                        help='Confidence threshold for pseudo-labels (default: 0.7)')
    parser.add_argument('--ramp_up_epochs', type=int, default=20,
                        help='Number of epochs to ramp up consistency weight (default: 20)')
    
    # Training configuration
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='vit-fish-id',
                        help='W&B project name (default: vit-fish-id)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detected if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    return parser.parse_args()


def validate_data_directory(data_dir: str, mode: str) -> Tuple[bool, str]:
    """Validate data directory structure."""
    if not os.path.exists(data_dir):
        return False, f"Directory does not exist: {data_dir}"
    
    if mode == 'semi_supervised':
        labeled_dir = os.path.join(data_dir, 'labeled')
        unlabeled_dir = os.path.join(data_dir, 'unlabeled')
        
        if not os.path.exists(labeled_dir):
            return False, f"Missing 'labeled' directory in {data_dir}"
        if not os.path.exists(unlabeled_dir):
            return False, f"Missing 'unlabeled' directory in {data_dir}"
    
    # Check if there are any subdirectories (class folders)
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(subdirs) == 0:
        return False, "No class directories found"
    
    return True, "Valid"


def setup_wandb(args, class_names: list, labeled_count: int = None, unlabeled_count: int = None):
    """Setup Weights & Biases logging."""
    if not args.use_wandb:
        return
    
    # Create experiment name
    if args.experiment_name:
        run_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.mode}_{args.model_name}_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            'model_name': args.model_name,
            'mode': args.mode,
            'num_classes': len(class_names),
            'class_names': class_names,
            'image_size': args.image_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'ema_momentum': args.ema_momentum,
            'consistency_weight': args.consistency_weight,
            'temperature': args.temperature,
            'labeled_samples': labeled_count,
            'unlabeled_samples': unlabeled_count,
            'device': args.device
        }
    )
    
    print(f"✅ W&B initialized: {args.wandb_project}/{run_name}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    if args.device is None:
        device = get_device()
    else:
        device = args.device
    
    print(f"🐟 ViT-FishID Training")
    print(f"📊 Mode: {args.mode}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Data directory: {args.data_dir}")
    
    # Validate data directory
    is_valid, error_msg = validate_data_directory(args.data_dir, args.mode)
    if not is_valid:
        print(f"❌ Invalid data directory: {error_msg}")
        
        if args.mode == 'semi_supervised':
            print(f"\n📋 Expected structure for semi-supervised training:")
            print(f"  {args.data_dir}/")
            print(f"  ├── labeled/")
            print(f"  │   ├── species_1/")
            print(f"  │   │   ├── img1.jpg")
            print(f"  │   │   └── img2.jpg")
            print(f"  │   └── species_2/")
            print(f"  │       └── ...")
            print(f"  └── unlabeled/")
            print(f"      ├── img3.jpg")
            print(f"      ├── img4.jpg")
            print(f"      └── ...")
        else:
            print(f"\n📋 Expected structure for supervised training:")
            print(f"  {args.data_dir}/")
            print(f"  ├── species_1/")
            print(f"  │   ├── img1.jpg")
            print(f"  │   └── img2.jpg")
            print(f"  ├── species_2/")
            print(f"  │   └── ...")
            print(f"  └── ...")
        
        sys.exit(1)
    
    # Create data loaders
    print(f"\n📦 Creating data loaders...")
    labeled_count = unlabeled_count = None
    
    if args.mode == 'semi_supervised':
        train_loader, val_loader, test_loader, class_names, labeled_count, unlabeled_count = create_semi_supervised_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            test_split=args.test_split,
            unlabeled_ratio=args.unlabeled_ratio,
            seed=args.seed
        )
    else:
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed
        )
        labeled_count = len(train_loader.dataset)
    
    num_classes = len(class_names)
    print(f"🏷️  Classes ({num_classes}): {class_names}")
    print(f"📊 Test set available with {len(test_loader.dataset):,} samples for final evaluation")
    
    # Create model and teacher
    print(f"\n🧠 Creating ViT model: {args.model_name}")
    student_model, ema_teacher = create_model_and_teacher(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate,
        ema_momentum=args.ema_momentum,
        device=device
    )
    
    # Print model info
    total_params = count_parameters(student_model)
    print(f"📊 Model parameters: {total_params:,}")
    
    # Setup wandb
    setup_wandb(args, class_names, labeled_count, unlabeled_count)
    
    # Create trainer
    print(f"\n🚀 Creating trainer...")
    if args.mode == 'semi_supervised':
        trainer = SemiSupervisedTrainer(
            student_model=student_model,
            ema_teacher=ema_teacher,
            num_classes=num_classes,
            device=device,
            consistency_weight=args.consistency_weight,
            pseudo_label_threshold=args.pseudo_label_threshold,
            temperature=args.temperature,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            ramp_up_epochs=args.ramp_up_epochs,
            use_wandb=args.use_wandb
        )
    else:
        trainer = EMATrainer(
            student_model=student_model,
            ema_teacher=ema_teacher,
            num_classes=num_classes,
            device=device,
            consistency_weight=args.consistency_weight,
            temperature=args.temperature,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            use_wandb=args.use_wandb
        )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"📥 Resuming from checkpoint: {args.resume_from}")
        # TODO: Implement checkpoint loading
        # load_checkpoint(args.resume_from, trainer)
    
    # Start training
    print(f"\n🎯 Starting {args.mode} training...")
    print(f"💡 Note: Test set is reserved for final evaluation and not used during training")
    train_model(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency
    )
    
    print(f"\n✅ Training completed!")
    print(f"💡 Use evaluate.py with the test set for final unbiased performance metrics")
    
    # Finish wandb
    if args.use_wandb:
        wandb.finish()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"💾 Checkpoints saved to: {args.save_dir}")
    print(f"🏆 Best accuracy: {trainer.best_accuracy:.2f}%")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        if wandb.run is not None:
            wandb.finish()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()
        sys.exit(1)
