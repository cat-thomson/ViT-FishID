"""
Unified Trainer Module for ViT-FishID.

This module contains:
- EMATrainer: Trainer for supervised learning with EMA teacher
- SemiSupervisedTrainer: Trainer for semi-supervised learning
- Utilities for training, validation, and checkpointing

Author: GitHub Copilot
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
import wandb
from tqdm import tqdm
import os

from model import ViTForFishClassification, EMATeacher, ConsistencyLoss
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint


class EMATrainer:
    """Trainer for supervised learning with EMA teacher-student framework."""
    
    def __init__(
        self,
        student_model: ViTForFishClassification,
        ema_teacher: EMATeacher,
        num_classes: int,
        device: str = 'cuda',
        consistency_weight: float = 1.0,
        temperature: float = 4.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        use_wandb: bool = True
    ):
        """
        Initialize supervised EMA trainer.
        
        Args:
            student_model: Student ViT model
            ema_teacher: EMA teacher wrapper
            num_classes: Number of classes
            device: Device to train on
            consistency_weight: Weight for consistency loss
            temperature: Temperature for consistency loss
            learning_rate: Initial learning rate
            weight_decay: Weight decay
            warmup_epochs: Number of warmup epochs
            use_wandb: Whether to use wandb logging
        """
        self.student_model = student_model
        self.ema_teacher = ema_teacher
        self.num_classes = num_classes
        self.device = device
        self.consistency_weight = consistency_weight
        self.warmup_epochs = warmup_epochs
        self.use_wandb = use_wandb
        
        # Loss functions
        self.supervised_loss = nn.CrossEntropyLoss()
        self.consistency_loss = ConsistencyLoss(temperature=temperature)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Tracking
        self.best_accuracy = 0.0
        self.current_epoch = 0
        
        print(f"✅ EMA Trainer initialized")
        print(f"  - Consistency weight: {consistency_weight}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Warmup epochs: {warmup_epochs}")
    
    def _warmup_learning_rate(self, epoch: int, batch_idx: int, total_batches: int):
        """Apply learning rate warmup."""
        if epoch < self.warmup_epochs:
            warmup_total_iters = self.warmup_epochs * total_batches
            current_iter = epoch * total_batches + batch_idx
            lr_scale = min(1.0, current_iter / warmup_total_iters)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer.defaults['lr'] * lr_scale
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch."""
        self.student_model.train()
        self.ema_teacher.teacher_model.eval()
        
        # Metrics
        supervised_losses = AverageMeter()
        consistency_losses = AverageMeter()
        total_losses = AverageMeter()
        top1_acc = AverageMeter()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)
            
            # Apply warmup
            self._warmup_learning_rate(epoch, batch_idx, len(train_loader))
            
            # Forward pass through student
            student_logits = self.student_model(images)
            
            # Supervised loss
            supervised_loss = self.supervised_loss(student_logits, targets)
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_logits = self.ema_teacher.teacher_model(images)
            
            # Consistency loss
            consistency_loss = self.consistency_loss(student_logits, teacher_logits)
            
            # Total loss
            total_loss = supervised_loss + self.consistency_weight * consistency_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update EMA teacher
            self.ema_teacher.update(self.student_model)
            
            # Update metrics
            acc1 = accuracy(student_logits, targets, topk=(1,))[0]
            supervised_losses.update(supervised_loss.item(), batch_size)
            consistency_losses.update(consistency_loss.item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)
            top1_acc.update(acc1.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_losses.avg:.4f}',
                'Sup': f'{supervised_losses.avg:.4f}',
                'Cons': f'{consistency_losses.avg:.4f}',
                'Acc': f'{top1_acc.avg:.2f}%'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train/supervised_loss': supervised_loss.item(),
                    'train/consistency_loss': consistency_loss.item(),
                    'train/total_loss': total_loss.item(),
                    'train/accuracy': acc1.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        return {
            'supervised_loss': supervised_losses.avg,
            'consistency_loss': consistency_losses.avg,
            'total_loss': total_losses.avg,
            'accuracy': top1_acc.avg
        }
    
    def validate(self, val_loader: DataLoader, use_teacher: bool = False) -> Dict[str, float]:
        """Validate model."""
        model = self.ema_teacher.teacher_model if use_teacher else self.student_model
        model.eval()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                batch_size = images.size(0)
                
                # Forward pass
                logits = model(images)
                loss = self.supervised_loss(logits, targets)
                
                # Calculate accuracy
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                
                # Update metrics
                losses.update(loss.item(), batch_size)
                top1_acc.update(acc1.item(), batch_size)
                top5_acc.update(acc5.item(), batch_size)
        
        return {
            'loss': losses.avg,
            'top1_accuracy': top1_acc.avg,
            'top5_accuracy': top5_acc.avg
        }


class SemiSupervisedTrainer:
    """Trainer for semi-supervised learning with EMA teacher-student framework."""
    
    def __init__(
        self,
        student_model: ViTForFishClassification,
        ema_teacher: EMATeacher,
        num_classes: int,
        device: str = 'cuda',
        consistency_weight: float = 2.0,
        pseudo_label_threshold: float = 0.7,
        temperature: float = 4.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        ramp_up_epochs: int = 20,
        use_wandb: bool = True
    ):
        """
        Initialize semi-supervised trainer.
        
        Args:
            student_model: Student ViT model
            ema_teacher: EMA teacher wrapper
            num_classes: Number of classes
            device: Device to train on
            consistency_weight: Weight for consistency loss
            pseudo_label_threshold: Confidence threshold for pseudo-labels
            temperature: Temperature for consistency loss
            learning_rate: Initial learning rate
            weight_decay: Weight decay
            warmup_epochs: Number of warmup epochs
            ramp_up_epochs: Number of epochs to ramp up consistency weight
            use_wandb: Whether to use wandb logging
        """
        self.student_model = student_model
        self.ema_teacher = ema_teacher
        self.num_classes = num_classes
        self.device = device
        self.consistency_weight = consistency_weight
        self.pseudo_label_threshold = pseudo_label_threshold
        self.warmup_epochs = warmup_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.use_wandb = use_wandb
        
        # Loss functions
        self.supervised_loss = nn.CrossEntropyLoss()
        self.consistency_loss = ConsistencyLoss(temperature=temperature)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Tracking
        self.best_accuracy = 0.0
        self.current_epoch = 0
        
        print(f"✅ Semi-Supervised Trainer initialized")
        print(f"  - Consistency weight: {consistency_weight}")
        print(f"  - Pseudo-label threshold: {pseudo_label_threshold}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Warmup epochs: {warmup_epochs}")
        print(f"  - Ramp-up epochs: {ramp_up_epochs}")
    
    def _get_consistency_weight(self, epoch: int) -> float:
        """Get current consistency weight with ramp-up."""
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.ramp_up_epochs:
            # Linear ramp-up
            ramp_progress = (epoch - self.warmup_epochs) / (self.ramp_up_epochs - self.warmup_epochs)
            return self.consistency_weight * ramp_progress
        else:
            return self.consistency_weight
    
    def _warmup_learning_rate(self, epoch: int, batch_idx: int, total_batches: int):
        """Apply learning rate warmup."""
        if epoch < self.warmup_epochs:
            warmup_total_iters = self.warmup_epochs * total_batches
            current_iter = epoch * total_batches + batch_idx
            lr_scale = min(1.0, current_iter / warmup_total_iters)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer.defaults['lr'] * lr_scale
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch with semi-supervised learning."""
        self.student_model.train()
        self.ema_teacher.teacher_model.eval()
        
        # Get current consistency weight
        current_consistency_weight = self._get_consistency_weight(epoch)
        
        # Metrics
        supervised_losses = AverageMeter()
        consistency_losses = AverageMeter()
        total_losses = AverageMeter()
        labeled_accuracy = AverageMeter()
        pseudo_label_accuracy = AverageMeter()
        
        # Counters
        labeled_samples = 0
        unlabeled_samples = 0
        high_conf_pseudo_labels = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets, is_labeled) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            is_labeled = is_labeled.to(self.device)
            batch_size = images.size(0)
            
            # Apply warmup
            self._warmup_learning_rate(epoch, batch_idx, len(train_loader))
            
            # Forward pass through student
            student_logits = self.student_model(images)
            
            # Forward pass through teacher (no gradients)
            with torch.no_grad():
                teacher_logits = self.ema_teacher.teacher_model(images)
                teacher_probs = F.softmax(teacher_logits, dim=1)
                max_probs, pseudo_targets = torch.max(teacher_probs, dim=1)
            
            # Supervised loss (only for labeled samples)
            labeled_mask = is_labeled.bool()
            supervised_loss = 0.0
            
            if labeled_mask.any():
                labeled_logits = student_logits[labeled_mask]
                labeled_targets = targets[labeled_mask]
                supervised_loss = self.supervised_loss(labeled_logits, labeled_targets)
                labeled_samples += labeled_mask.sum().item()
                
                # Calculate labeled accuracy
                labeled_acc = accuracy(labeled_logits, labeled_targets, topk=(1,))[0]
                labeled_accuracy.update(labeled_acc.item(), labeled_mask.sum().item())
            
            # Consistency loss (for ALL unlabeled samples, not just high-confidence)
            unlabeled_mask = ~labeled_mask
            consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            if unlabeled_mask.any():
                unlabeled_student_logits = student_logits[unlabeled_mask]
                unlabeled_teacher_logits = teacher_logits[unlabeled_mask]
                
                # Apply consistency loss to ALL unlabeled samples
                consistency_loss = self.consistency_loss(unlabeled_student_logits, unlabeled_teacher_logits)
                unlabeled_samples += unlabeled_mask.sum().item()
                
                # Track high-confidence pseudo-labels for monitoring
                unlabeled_max_probs = max_probs[unlabeled_mask]
                unlabeled_pseudo_targets = pseudo_targets[unlabeled_mask]
                high_conf_mask = unlabeled_max_probs > self.pseudo_label_threshold
                high_conf_pseudo_labels += high_conf_mask.sum().item()
                
                # Calculate pseudo-label accuracy for monitoring
                if high_conf_mask.any():
                    pseudo_acc = accuracy(
                        unlabeled_student_logits[high_conf_mask], 
                        unlabeled_pseudo_targets[high_conf_mask], 
                        topk=(1,)
                    )[0]
                    pseudo_label_accuracy.update(pseudo_acc.item(), high_conf_mask.sum().item())
            
            # Total loss
            total_loss = supervised_loss + current_consistency_weight * consistency_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update EMA teacher
            self.ema_teacher.update(self.student_model)
            
            # Update metrics
            if isinstance(supervised_loss, torch.Tensor):
                supervised_losses.update(supervised_loss.item(), batch_size)
            if isinstance(consistency_loss, torch.Tensor):
                consistency_losses.update(consistency_loss.item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{total_losses.avg:.4f}',
                'Sup': f'{supervised_losses.avg:.4f}',
                'Cons': f'{consistency_losses.avg:.4f}',
                'L-Acc': f'{labeled_accuracy.avg:.1f}%',
                'P-Acc': f'{pseudo_label_accuracy.avg:.1f}%'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train/supervised_loss': supervised_loss.item() if isinstance(supervised_loss, torch.Tensor) else 0,
                    'train/consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0,
                    'train/total_loss': total_loss.item(),
                    'train/labeled_accuracy': labeled_accuracy.avg,
                    'train/pseudo_accuracy': pseudo_label_accuracy.avg,
                    'train/consistency_weight': current_consistency_weight,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/high_conf_ratio': high_conf_pseudo_labels / max(1, unlabeled_samples) * 100,
                    'epoch': epoch
                })
        
        return {
            'supervised_loss': supervised_losses.avg,
            'consistency_loss': consistency_losses.avg,
            'total_loss': total_losses.avg,
            'labeled_accuracy': labeled_accuracy.avg,
            'pseudo_accuracy': pseudo_label_accuracy.avg,
            'high_conf_pseudo_labels': high_conf_pseudo_labels,
            'labeled_samples': labeled_samples,
            'unlabeled_samples': unlabeled_samples
        }
    
    def validate(self, val_loader: DataLoader, use_teacher: bool = False) -> Dict[str, float]:
        """Validate model."""
        model = self.ema_teacher.teacher_model if use_teacher else self.student_model
        model.eval()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                batch_size = images.size(0)
                
                # Forward pass
                logits = model(images)
                loss = self.supervised_loss(logits, targets)
                
                # Calculate accuracy
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                
                # Update metrics
                losses.update(loss.item(), batch_size)
                top1_acc.update(acc1.item(), batch_size)
                top5_acc.update(acc5.item(), batch_size)
        
        return {
            'loss': losses.avg,
            'top1_accuracy': top1_acc.avg,
            'top5_accuracy': top5_acc.avg
        }


def train_model(
    trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    save_dir: str = './checkpoints',
    save_frequency: int = 10,
    start_epoch: int = 1
):
    """
    Full training loop for both supervised and semi-supervised training.
    
    Args:
        trainer: EMATrainer or SemiSupervisedTrainer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        save_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N epochs
        start_epoch: Epoch to start training from (for resuming)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize scheduler (adjust for resume)
    if start_epoch > 1:
        # For resumed training, we need to adjust the scheduler
        remaining_epochs = epochs - start_epoch + 1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=remaining_epochs, eta_min=1e-6
        )
        print(f"🔄 Resuming training from epoch {start_epoch}")
        print(f"⏰ Remaining epochs: {remaining_epochs}")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=epochs, eta_min=1e-6
        )
        print(f"🚀 Starting fresh training for {epochs} epochs")
    
    print(f"📁 Checkpoints will be saved to: {save_dir}")
    
    for epoch in range(start_epoch, epochs + 1):
        trainer.current_epoch = epoch
        
        # Training
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        
        # Validation (both student and teacher)
        student_val_metrics = trainer.validate(val_loader, use_teacher=False)
        teacher_val_metrics = trainer.validate(val_loader, use_teacher=True)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        print(f"\n📊 Epoch {epoch+1}/{epochs}")
        print(f"Train - Total Loss: {train_metrics['total_loss']:.4f}")
        if 'labeled_accuracy' in train_metrics:
            # Semi-supervised metrics
            print(f"Train - Labeled Acc: {train_metrics['labeled_accuracy']:.1f}%, Pseudo Acc: {train_metrics['pseudo_accuracy']:.1f}%")
            print(f"Train - High-conf Pseudo: {train_metrics['high_conf_pseudo_labels']}/{train_metrics['unlabeled_samples']} ({train_metrics['high_conf_pseudo_labels']/max(1, train_metrics['unlabeled_samples'])*100:.1f}%)")
        else:
            # Supervised metrics
            print(f"Train - Accuracy: {train_metrics['accuracy']:.1f}%")
        
        print(f"Student Val - Acc: {student_val_metrics['top1_accuracy']:.1f}%")
        print(f"Teacher Val - Acc: {teacher_val_metrics['top1_accuracy']:.1f}%")
        
        # Log to wandb
        if trainer.use_wandb:
            log_dict = {
                'epoch': epoch,
                'val/student_top1_acc': student_val_metrics['top1_accuracy'],
                'val/student_top5_acc': student_val_metrics['top5_accuracy'], 
                'val/teacher_top1_acc': teacher_val_metrics['top1_accuracy'],
                'val/teacher_top5_acc': teacher_val_metrics['top5_accuracy'],
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            # Add training metrics
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    log_dict[f'train_epoch/{key}'] = value
            
            wandb.log(log_dict)
        
        # Save checkpoint - EVERY EPOCH with Google Drive backup
        is_best = teacher_val_metrics['top1_accuracy'] > trainer.best_accuracy
        if is_best:
            trainer.best_accuracy = teacher_val_metrics['top1_accuracy']
            print(f"🏆 New best accuracy: {trainer.best_accuracy:.2f}%")
        
        # Save checkpoint every epoch (changed from save_frequency check)
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': trainer.student_model.state_dict(),
            'ema_teacher_state_dict': trainer.ema_teacher.teacher_model.state_dict(),  # Fixed key name
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_accuracy': trainer.best_accuracy,
            'train_metrics': train_metrics,
            'val_metrics': teacher_val_metrics,
            'num_classes': getattr(trainer, 'num_classes', 37),
            'consistency_weight': getattr(trainer, 'consistency_weight', 2.0),
            'pseudo_label_threshold': getattr(trainer, 'pseudo_label_threshold', 0.7)
        }
        
        # Save regular checkpoint every epoch
        checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(checkpoint, is_best, save_dir, checkpoint_filename)
        
        # Additional Google Drive backup every 5 epochs
        if epoch % 5 == 0 or is_best:
            # Try to save to Google Drive backup location
            try:
                google_drive_backup = '/content/drive/MyDrive/ViT-FishID/checkpoints_backup'
                if save_dir.startswith('/content/drive/MyDrive'):
                    # Already saving to Google Drive, create additional backup
                    backup_dir = google_drive_backup
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, checkpoint_filename)
                    torch.save(checkpoint, backup_path)
                    print(f"💾 Backup saved to: {backup_path}")
                elif not save_dir.startswith('/content/drive/MyDrive'):
                    # Local training, need to save to Google Drive
                    print(f"💾 Saving epoch {epoch} backup to Google Drive...")
                    backup_dir = google_drive_backup
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, checkpoint_filename)
                    torch.save(checkpoint, backup_path)
                    print(f"✅ Google Drive backup: {backup_path}")
            except Exception as e:
                print(f"⚠️ Could not save Google Drive backup: {e}")
        
        print(f"📊 Epoch {epoch} checkpoint saved (Size: {os.path.getsize(os.path.join(save_dir, checkpoint_filename)) / (1024*1024):.1f} MB)")
    
    print(f"\n🎉 Training completed! Best validation accuracy: {trainer.best_accuracy:.2f}%")
