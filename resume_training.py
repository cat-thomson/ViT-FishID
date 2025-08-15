"""
Resume Training Script for ViT-FishID in Google Colab.

This script helps you resume training from a checkpoint if your Colab session times out.
Add this to your Colab notebook as a new cell.
"""

import torch
import os
from model import ViTForFishClassification, EMATeacher
from trainer import SemiSupervisedTrainer
from utils import load_checkpoint

def resume_training_from_checkpoint(
    checkpoint_path: str,
    labeled_loader,
    unlabeled_loader, 
    val_loader,
    test_loader,
    num_classes: int,
    device: str = 'cuda',
    remaining_epochs: int = None
):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        labeled_loader: DataLoader for labeled data
        unlabeled_loader: DataLoader for unlabeled data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        num_classes: Number of classes
        device: Device to use for training
        remaining_epochs: Number of remaining epochs (if None, will calculate automatically)
    """
    
    print(f"🔄 Resuming training from: {checkpoint_path}")
    
    # Initialize models
    student = ViTForFishClassification(
        num_classes=num_classes,
        model_name='vit_base_patch16_224',
        pretrained=True
    ).to(device)
    
    ema_teacher = EMATeacher(
        student_model=student,
        decay=0.999,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model states
    student.load_state_dict(checkpoint['student_state_dict'])
    ema_teacher.teacher_model.load_state_dict(checkpoint['ema_teacher_state_dict'])
    
    # Get the last completed epoch
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"📊 Last completed epoch: {checkpoint['epoch']}")
    print(f"📊 Best validation accuracy so far: {best_val_acc:.2f}%")
    print(f"🚀 Resuming from epoch: {start_epoch}")
    
    # Calculate remaining epochs
    if remaining_epochs is None:
        total_epochs = 50  # Your original total
        remaining_epochs = total_epochs - start_epoch + 1
    
    print(f"⏰ Remaining epochs: {remaining_epochs}")
    
    # Initialize trainer
    trainer = SemiSupervisedTrainer(
        student_model=student,
        ema_teacher=ema_teacher,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        use_wandb=True
    )
    
    # Resume training
    trainer.train(
        num_epochs=remaining_epochs,
        save_dir=os.path.dirname(checkpoint_path),
        save_frequency=5,  # Save every 5 epochs
        patience=15,
        start_epoch=start_epoch,  # Start from the next epoch
        best_val_acc=best_val_acc  # Continue tracking best accuracy
    )
    
    return trainer

# Function to find the latest checkpoint
def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    latest = checkpoints[-1]
    
    return os.path.join(checkpoint_dir, latest)

# Example usage for Colab:
"""
# In your Colab notebook, add this cell:

checkpoint_dir = '/content/drive/MyDrive/ViT-FishID/checkpoints'
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    print(f"Found checkpoint: {latest_checkpoint}")
    
    # Resume training
    trainer = resume_training_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device
    )
else:
    print("No checkpoint found. Starting fresh training.")
    # Start normal training
"""
