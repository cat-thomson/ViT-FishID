#!/usr/bin/env python3
"""
Main training script for ViT-Base with EMA Teacher-Student framework.

This script implements an Exponential Moving Average (EMA) teacher-student training
framework for Vision Transformer (ViT) models on fish classification.

Usage:
    python main.py --data_dir /path/to/fish/dataset --epochs 100 --batch_size 32

Author: GitHub Copilot
Date: 2025
"""

import argparse
import os
import sys
import torch
import wandb
from datetime import datetime

# Import custom modules
from vit_model import ViTForFishClassification
from ema_trainer import EMATrainer
from data_loader import create_fish_dataloaders
from utils import set_seed, get_device, count_parameters


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ViT with EMA Teacher-Student Framework')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to fish dataset directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    
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
                        help='EMA momentum for teacher updates (default: 0.999)')
    parser.add_argument('--consistency_loss', type=str, default='mse', choices=['mse', 'kl'],
                        help='Type of consistency loss (default: mse)')
    parser.add_argument('--consistency_weight', type=float, default=1.0,
                        help='Weight for consistency loss (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for consistency loss (default: 4.0)')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu). If None, auto-detect')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='vit-fish-ema',
                        help='Wandb project name (default: vit-fish-ema)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def setup_wandb(args, class_names):
    """Setup Weights & Biases logging."""
    if not args.use_wandb:
        return
    
    # Generate run name if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.wandb_run_name = f"ema_vit_{args.model_name}_{timestamp}"
    
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
            'temperature': args.temperature,
            'num_classes': len(class_names),
            'class_names': class_names,
            'pretrained': args.pretrained,
            'dropout_rate': args.dropout_rate,
            'warmup_epochs': args.warmup_epochs
        }
    )
    
    print(f"Wandb initialized: {args.wandb_project}/{args.wandb_run_name}")


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
    
    print(f"Starting ViT-Base EMA Teacher-Student Training")
    print(f"Arguments: {vars(args)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, class_names = create_fish_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create model
    print(f"\nCreating ViT model: {args.model_name}")
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
    setup_wandb(args, class_names)
    
    # Create trainer
    print(f"\nInitializing EMA Trainer...")
    trainer = EMATrainer(
        student_model=student_model,
        num_classes=num_classes,
        device=device,
        ema_momentum=args.ema_momentum,
        consistency_loss_type=args.consistency_loss,
        consistency_weight=args.consistency_weight,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Implementation would load checkpoint here
        pass
    
    # Start training
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency
    )
    
    print("Training completed!")
    
    # Finish wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)
