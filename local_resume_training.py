#!/usr/bin/env python3
"""
Local Training Script for ViT-FishID - Resume from Epoch 19

This script is specifically configured to:
1. Resume training from epoch 19 checkpoint
2. Save checkpoints at every epoch
3. Run locally in VS Code
4. Train for 100 total epochs (81 remaining)

Usage:
    python local_resume_training.py

Author: GitHub Copilot
Date: 2025
"""

import os
import sys
import torch
import json
import shutil
from datetime import datetime
from pathlib import Path

# Import unified modules
from model import create_model_and_teacher
from trainer import SemiSupervisedTrainer
from data import create_semi_supervised_dataloaders
from utils import set_seed, get_device, count_parameters


class LocalTrainingConfig:
    """Configuration for local training."""
    
    def __init__(self):
        # Paths
        self.project_root = Path("/Users/catalinathomson/Desktop/Fish/ViT-FishID")
        self.data_dir = self.project_root / "fish_cutouts"
        self.checkpoint_path = self.project_root / "checkpoint_epoch_19.pth"
        self.save_dir = self.project_root / "local_checkpoints"
        
        # Resume settings
        self.start_epoch = 20  # Next epoch after 19
        self.total_epochs = 100
        self.remaining_epochs = 81  # 100 - 19
        
        # Training settings
        self.mode = 'semi_supervised'
        self.batch_size = 8  # Reduced for local GPU
        self.learning_rate = 1e-4
        self.weight_decay = 0.05
        
        # Model settings
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = 37  # Will be auto-detected
        self.pretrained = True
        self.dropout_rate = 0.1
        
        # Semi-supervised settings
        self.consistency_weight = 2.0
        self.pseudo_label_threshold = 0.7
        self.temperature = 4.0
        self.warmup_epochs = 5
        self.ramp_up_epochs = 15
        self.ema_momentum = 0.999
        self.unlabeled_ratio = 8.0  # Increased to use more unlabeled data (was 2.0)
        
        # Checkpoint settings - SAVE EVERY EPOCH
        self.save_frequency = 1
        
        # Data settings
        self.image_size = 224
        self.num_workers = 2  # Reduced for local
        self.val_split = 0.1  # Reduced to use more data for training
        self.test_split = 0.1  # Reduced to use more data for training
        
        # System settings
        self.seed = 42
        self.device = None  # Auto-detect
        
        # Logging
        self.use_wandb = False  # Disabled for local training


def validate_environment(config):
    """Validate the training environment."""
    print("🔍 VALIDATING TRAINING ENVIRONMENT")
    print("="*50)
    
    # Check data directory
    if not config.data_dir.exists():
        print(f"❌ Data directory not found: {config.data_dir}")
        return False
    
    labeled_dir = config.data_dir / "labeled"
    unlabeled_dir = config.data_dir / "unlabeled"
    
    if not labeled_dir.exists():
        print(f"❌ Labeled directory not found: {labeled_dir}")
        return False
    
    if not unlabeled_dir.exists():
        print(f"❌ Unlabeled directory not found: {unlabeled_dir}")
        return False
    
    # Count data
    species_folders = [d for d in labeled_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    unlabeled_files = [f for f in unlabeled_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"✅ Found {len(species_folders)} species in labeled data")
    print(f"✅ Found {len(unlabeled_files)} unlabeled images")
    
    config.num_classes = len(species_folders)
    
    # Check checkpoint
    if not config.checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {config.checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        epoch = checkpoint.get('epoch', 'Unknown')
        accuracy = checkpoint.get('best_accuracy', checkpoint.get('best_acc', 'Unknown'))
        print(f"✅ Checkpoint loaded: Epoch {epoch}, Accuracy: {accuracy}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
    
    # Check GPU
    device = get_device()
    print(f"✅ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create save directory
    config.save_dir.mkdir(exist_ok=True)
    print(f"✅ Save directory: {config.save_dir}")
    
    return True


def load_checkpoint_and_resume(config, trainer):
    """Load checkpoint and prepare for resuming."""
    print(f"\n📂 LOADING CHECKPOINT FROM EPOCH 19")
    print("="*50)
    
    checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
    
    # Load model states
    if 'student_state_dict' in checkpoint:
        trainer.student_model.load_state_dict(checkpoint['student_state_dict'])
        print("✅ Student model state loaded")
    else:
        print("⚠️ Student state dict not found in checkpoint")
    
    if 'teacher_state_dict' in checkpoint:
        trainer.ema_teacher.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        print("✅ Teacher model state loaded")
    elif 'ema_teacher_state_dict' in checkpoint:
        trainer.ema_teacher.teacher_model.load_state_dict(checkpoint['ema_teacher_state_dict'])
        print("✅ Teacher model state loaded (from ema_teacher_state_dict)")
    else:
        print("⚠️ Teacher state dict not found in checkpoint")
    
    # Load optimizer
    if 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded")
    
    # Get training info
    start_epoch = checkpoint.get('epoch', 19) + 1
    best_accuracy = checkpoint.get('best_accuracy', checkpoint.get('best_acc', 0.0))
    trainer.best_accuracy = best_accuracy
    trainer.current_epoch = start_epoch - 1
    
    print(f"📊 Resuming from epoch: {start_epoch}")
    print(f"📊 Best accuracy so far: {best_accuracy:.2f}%")
    
    return start_epoch, best_accuracy



def create_training_summary(config, results):
    """Create a training summary file."""
    summary_path = config.save_dir / "training_summary.json"
    
    summary = {
        "training_info": {
            "start_time": results.get('start_time', ''),
            "end_time": results.get('end_time', ''),
            "total_epochs": config.total_epochs,
            "epochs_completed": results.get('epochs_completed', 0),
            "best_epoch": results.get('best_epoch', 0),
            "best_accuracy": results.get('best_accuracy', 0.0)
        },
        "config": vars(config),
        "final_metrics": results.get('final_metrics', {})
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Training summary saved: {summary_path}")


def main():
    """Main training function."""
    print("🚀 ViT-FishID LOCAL TRAINING - RESUME FROM EPOCH 19")
    print("="*60)
    
    # Initialize configuration
    config = LocalTrainingConfig()
    
    # Validate environment
    if not validate_environment(config):
        print("❌ Environment validation failed. Please fix the issues above.")
        return
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    config.device = device
    
    print(f"\n🎯 TRAINING CONFIGURATION")
    print("="*50)
    print(f"📊 Resume from: Epoch 19")
    print(f"📊 Target epochs: {config.total_epochs}")
    print(f"📊 Remaining epochs: {config.remaining_epochs}")
    print(f"📊 Batch size: {config.batch_size}")
    print(f"📊 Learning rate: {config.learning_rate}")
    print(f"📊 Save frequency: Every epoch")
    print(f"📊 Device: {device}")
    
    # Create data loaders
    print(f"\n📊 CREATING DATA LOADERS")
    print("="*50)
    
    train_loader, val_loader, test_loader, class_names, labeled_count, unlabeled_count = create_semi_supervised_dataloaders(
        data_dir=str(config.data_dir),
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        test_split=config.test_split,
        unlabeled_ratio=config.unlabeled_ratio
    )
    
    print(f"✅ Created data loaders")
    print(f"   - Classes: {len(class_names)}")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    print(f"   - Labeled samples: {labeled_count}")
    print(f"   - Unlabeled samples: {unlabeled_count}")
    
    # Create models
    print(f"\n🏗️ CREATING MODELS")
    print("="*50)
    
    model, teacher_model = create_model_and_teacher(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        ema_momentum=config.ema_momentum,
        device=device
    )
    
    print(f"✅ Models created")
    print(f"   - Parameters: {count_parameters(model):,}")
    print(f"   - Model: {config.model_name}")
    
    # Create trainer
    trainer = SemiSupervisedTrainer(
        student_model=model,
        ema_teacher=teacher_model,
        num_classes=config.num_classes,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        consistency_weight=config.consistency_weight,
        temperature=config.temperature,
        warmup_epochs=config.warmup_epochs,
        ramp_up_epochs=config.ramp_up_epochs,
        pseudo_label_threshold=config.pseudo_label_threshold,
        use_wandb=config.use_wandb
    )
    
    # Load checkpoint and resume
    start_epoch, best_accuracy = load_checkpoint_and_resume(config, trainer)
    
    # Start training using the trainer's train_model function
    print(f"\n🎬 STARTING TRAINING - EPOCH {start_epoch} TO {config.total_epochs}")
    print("="*60)
    
    start_time = datetime.now()
    results = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_accuracy': best_accuracy,
        'best_epoch': start_epoch - 1
    }
    
    try:
        # Import the train_model function
        from trainer import train_model
        
        # Run training
        train_model(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.total_epochs,
            save_dir=str(config.save_dir),
            save_frequency=config.save_frequency,
            start_epoch=start_epoch
        )
        
        # Training completed successfully
        final_epoch = config.total_epochs
        
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        final_epoch = trainer.current_epoch
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        final_epoch = trainer.current_epoch if hasattr(trainer, 'current_epoch') else start_epoch
    
    finally:
        # Training completed
        end_time = datetime.now()
        results['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
        results['epochs_completed'] = final_epoch
        results['best_accuracy'] = trainer.best_accuracy
        
        # Find the best epoch by checking saved checkpoints
        best_epoch = start_epoch - 1
        if hasattr(trainer, 'current_epoch'):
            best_epoch = trainer.current_epoch
        results['best_epoch'] = best_epoch
        
        # Create training summary
        create_training_summary(config, results)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print("="*60)
        print(f"⏱️ Start time: {results['start_time']}")
        print(f"⏱️ End time: {results['end_time']}")
        print(f"⏱️ Duration: {end_time - start_time}")
        print(f"📊 Epochs completed: {results['epochs_completed']}")
        print(f"🏆 Best accuracy: {results['best_accuracy']:.2f}% (Epoch {results['best_epoch']})")
        print(f"💾 Checkpoints saved to: {config.save_dir}")
        
        # Final evaluation on test set
        print(f"\n🧪 FINAL TEST EVALUATION")
        print("-" * 30)
        test_metrics = trainer.validate(test_loader)
        print(f"📊 Test Loss: {test_metrics['loss']:.4f}")
        print(f"📊 Test Accuracy: {test_metrics['top1_accuracy']:.2f}%")
        
        results['final_metrics'] = {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['top1_accuracy']
        }
        
        print(f"\n✅ All checkpoints saved to: {config.save_dir}")
        print(f"✅ Use model_best.pth for inference")
        print(f"✅ Training summary: {config.save_dir}/training_summary.json")


if __name__ == "__main__":
    main()
