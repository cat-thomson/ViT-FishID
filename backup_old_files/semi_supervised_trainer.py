import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
import wandb
from tqdm import tqdm

from vit_model import ViTForFishClassification
from ema_teacher import EMATeacher, ConsistencyLoss, KLDivergenceLoss
from semi_supervised_data import create_semi_supervised_dataloaders
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint


class SemiSupervisedEMATrainer:
    """
    Semi-supervised trainer for EMA Teacher-Student framework with ViT.
    Handles both labeled and unlabeled data.
    """
    
    def __init__(
        self,
        student_model: ViTForFishClassification,
        num_classes: int,
        device: str = 'cuda',
        ema_momentum: float = 0.999,
        consistency_loss_type: str = 'mse',
        consistency_weight: float = 1.0,
        pseudo_label_threshold: float = 0.95,
        temperature: float = 4.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        ramp_up_epochs: int = 20,
        use_wandb: bool = True
    ):
        """
        Initialize Semi-Supervised EMA Trainer.
        
        Args:
            student_model: Student ViT model
            num_classes: Number of classes
            device: Training device
            ema_momentum: EMA momentum for teacher updates
            consistency_loss_type: 'mse' or 'kl' for consistency loss
            consistency_weight: Weight for consistency loss
            pseudo_label_threshold: Confidence threshold for pseudo-labels
            temperature: Temperature for softmax in consistency loss
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            ramp_up_epochs: Epochs to ramp up consistency weight
            use_wandb: Whether to use Weights & Biases logging
        """
        self.device = device
        self.num_classes = num_classes
        self.consistency_weight = consistency_weight
        self.pseudo_label_threshold = pseudo_label_threshold
        self.temperature = temperature
        self.consistency_loss_type = consistency_loss_type
        self.warmup_epochs = warmup_epochs
        self.ramp_up_epochs = ramp_up_epochs
        self.use_wandb = use_wandb
        
        # Models
        self.student_model = student_model.to(device)
        self.ema_teacher = EMATeacher(student_model, momentum=ema_momentum, device=device)
        
        # Loss functions
        self.supervised_loss = nn.CrossEntropyLoss()
        
        if consistency_loss_type == 'mse':
            self.consistency_loss = ConsistencyLoss(temperature=temperature, alpha=1.0)
        elif consistency_loss_type == 'kl':
            self.consistency_loss = KLDivergenceLoss(temperature=temperature, alpha=1.0)
        else:
            raise ValueError(f"Invalid consistency loss type: {consistency_loss_type}")
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Metrics tracking
        self.best_accuracy = 0.0
        self.current_epoch = 0
        
        print(f"Semi-Supervised EMA Trainer initialized:")
        print(f"  - EMA momentum: {ema_momentum}")
        print(f"  - Consistency loss: {consistency_loss_type}")
        print(f"  - Consistency weight: {consistency_weight}")
        print(f"  - Pseudo-label threshold: {pseudo_label_threshold}")
        print(f"  - Temperature: {temperature}")
    
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
                param_group['lr'] = param_group['lr'] * lr_scale
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Train for one epoch with semi-supervised learning.
        
        Args:
            train_loader: DataLoader with both labeled and unlabeled data
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
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
        high_confidence_pseudo_labels = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} (λ={current_consistency_weight:.3f})')
        
        for batch_idx, (images, labels, is_labeled) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            is_labeled = is_labeled.to(self.device)
            batch_size = images.size(0)
            
            # Apply warmup
            self._warmup_learning_rate(epoch, batch_idx, len(train_loader))
            
            # Separate labeled and unlabeled samples
            labeled_mask = is_labeled.bool()
            unlabeled_mask = ~labeled_mask
            
            labeled_images = images[labeled_mask]
            labeled_labels = labels[labeled_mask]
            unlabeled_images = images[unlabeled_mask]
            
            total_loss = 0.0
            supervised_loss = torch.tensor(0.0).to(self.device)
            consistency_loss = torch.tensor(0.0).to(self.device)
            
            # Supervised loss on labeled data
            if labeled_images.size(0) > 0:
                student_labeled_logits = self.student_model(labeled_images)
                supervised_loss = self.supervised_loss(student_labeled_logits, labeled_labels)
                total_loss += supervised_loss
                
                # Calculate labeled accuracy
                labeled_acc = accuracy(student_labeled_logits, labeled_labels, topk=(1,))[0]
                labeled_accuracy.update(labeled_acc.item(), labeled_images.size(0))
                labeled_samples += labeled_images.size(0)
            
                # Consistency loss on unlabeled data
            if unlabeled_images.size(0) > 0 and current_consistency_weight > 0:
                # Student prediction on unlabeled data
                student_unlabeled_logits = self.student_model(unlabeled_images)
                
                # Teacher prediction on unlabeled data (no gradients)
                with torch.no_grad():
                    teacher_unlabeled_logits = self.ema_teacher.teacher_model(unlabeled_images)
                    teacher_probs = torch.softmax(teacher_unlabeled_logits / self.temperature, dim=1)
                    
                    # Generate pseudo-labels from teacher
                    max_probs, pseudo_labels = torch.max(teacher_probs, dim=1)
                    
                    # Count high-confidence pseudo-labels
                    high_confidence_mask = max_probs > self.pseudo_label_threshold
                    high_confidence_pseudo_labels += high_confidence_mask.sum().item()
                    
                    # Calculate pseudo-label accuracy if we had true labels
                    # (This is just for monitoring - we don't use true labels in training)
                    student_pred = torch.argmax(student_unlabeled_logits, dim=1)
                    pseudo_acc = (student_pred == pseudo_labels).float().mean() * 100
                    pseudo_label_accuracy.update(pseudo_acc.item(), unlabeled_images.size(0))
                
                # Apply consistency loss to ALL unlabeled data (not just high-confidence)
                # This is the key fix - consistency loss should always be computed
                if self.consistency_loss_type == 'mse':
                    # MSE between softmax probabilities
                    student_probs = torch.softmax(student_unlabeled_logits / self.temperature, dim=1)
                    teacher_probs_detached = teacher_probs.detach()
                    consistency_loss = torch.mean((student_probs - teacher_probs_detached) ** 2)
                else:
                    # KL divergence consistency loss
                    consistency_loss = self.consistency_loss(
                        torch.log_softmax(student_unlabeled_logits / self.temperature, dim=1),
                        teacher_probs.detach()
                    )
                
                total_loss += current_consistency_weight * consistency_loss
                unlabeled_samples += unlabeled_images.size(0)            # Backward pass
            self.optimizer.zero_grad()
            if total_loss > 0:
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Update EMA teacher
            self.ema_teacher.update(self.student_model)
            
            # Update metrics
            supervised_losses.update(supervised_loss.item(), batch_size)
            consistency_losses.update(consistency_loss.item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_losses.avg:.4f}',
                'Sup': f'{supervised_losses.avg:.4f}',
                'Cons': f'{consistency_losses.avg:.4f}',
                'L_Acc': f'{labeled_accuracy.avg:.1f}%',
                'P_Acc': f'{pseudo_label_accuracy.avg:.1f}%'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train/supervised_loss': supervised_loss.item(),
                    'train/consistency_loss': consistency_loss.item(),
                    'train/total_loss': total_loss.item(),
                    'train/labeled_accuracy': labeled_accuracy.avg,
                    'train/pseudo_label_accuracy': pseudo_label_accuracy.avg,
                    'train/consistency_weight': current_consistency_weight,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        # Calculate final statistics
        pseudo_label_usage = high_confidence_pseudo_labels / max(unlabeled_samples, 1) * 100
        
        return {
            'supervised_loss': supervised_losses.avg,
            'consistency_loss': consistency_losses.avg,
            'total_loss': total_losses.avg,
            'labeled_accuracy': labeled_accuracy.avg,
            'pseudo_label_accuracy': pseudo_label_accuracy.avg,
            'labeled_samples': labeled_samples,
            'unlabeled_samples': unlabeled_samples,
            'pseudo_label_usage': pseudo_label_usage,
            'consistency_weight': current_consistency_weight
        }
    
    def validate(self, val_loader: DataLoader, use_teacher: bool = False) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            use_teacher: Whether to use teacher model for validation
            
        Returns:
            Dictionary of validation metrics
        """
        model = self.ema_teacher.teacher_model if use_teacher else self.student_model
        model.eval()
        
        losses = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        
        with torch.no_grad():
            for images, targets, is_labeled in tqdm(val_loader, desc='Validating'):
                # Only use labeled samples for validation
                labeled_mask = is_labeled.bool()
                if not labeled_mask.any():
                    continue
                
                images = images[labeled_mask].to(self.device)
                targets = targets[labeled_mask].to(self.device)
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_dir: str = './checkpoints',
        save_frequency: int = 10
    ):
        """
        Full semi-supervised training loop.
        
        Args:
            train_loader: Training data loader (labeled + unlabeled)
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N epochs
        """
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        
        print(f"Starting semi-supervised training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.student_model.parameters()):,}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch=epoch)
            
            # Validation (both student and teacher)
            student_val_metrics = self.validate(val_loader, use_teacher=False)
            teacher_val_metrics = self.validate(val_loader, use_teacher=True)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train - Total Loss: {train_metrics['total_loss']:.4f}, " +
                  f"Sup Loss: {train_metrics['supervised_loss']:.4f}, " +
                  f"Cons Loss: {train_metrics['consistency_loss']:.4f}")
            print(f"Train - Labeled Acc: {train_metrics['labeled_accuracy']:.2f}%, " +
                  f"Pseudo Acc: {train_metrics['pseudo_label_accuracy']:.2f}%")
            print(f"Train - Labeled: {train_metrics['labeled_samples']}, " +
                  f"Unlabeled: {train_metrics['unlabeled_samples']}, " +
                  f"High-conf Pseudo: {train_metrics['pseudo_label_usage']:.1f}%")
            print(f"Student Val - Loss: {student_val_metrics['loss']:.4f}, " +
                  f"Acc: {student_val_metrics['top1_accuracy']:.2f}%")
            print(f"Teacher Val - Loss: {teacher_val_metrics['loss']:.4f}, " +
                  f"Acc: {teacher_val_metrics['top1_accuracy']:.2f}%")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_total_loss': train_metrics['total_loss'],
                    'train/epoch_supervised_loss': train_metrics['supervised_loss'],
                    'train/epoch_consistency_loss': train_metrics['consistency_loss'],
                    'train/epoch_labeled_accuracy': train_metrics['labeled_accuracy'],
                    'train/epoch_pseudo_accuracy': train_metrics['pseudo_label_accuracy'],
                    'train/labeled_samples': train_metrics['labeled_samples'],
                    'train/unlabeled_samples': train_metrics['unlabeled_samples'],
                    'train/pseudo_label_usage': train_metrics['pseudo_label_usage'],
                    'train/consistency_weight': train_metrics['consistency_weight'],
                    'val/student_loss': student_val_metrics['loss'],
                    'val/student_top1_acc': student_val_metrics['top1_accuracy'],
                    'val/student_top5_acc': student_val_metrics['top5_accuracy'],
                    'val/teacher_loss': teacher_val_metrics['loss'],
                    'val/teacher_top1_acc': teacher_val_metrics['top1_accuracy'],
                    'val/teacher_top5_acc': teacher_val_metrics['top5_accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = teacher_val_metrics['top1_accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = teacher_val_metrics['top1_accuracy']
            
            if epoch % save_frequency == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'student_state_dict': self.student_model.state_dict(),
                    'teacher_state_dict': self.ema_teacher.teacher_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_accuracy': self.best_accuracy,
                    'train_metrics': train_metrics,
                    'val_metrics': teacher_val_metrics
                }, is_best, save_dir, f'checkpoint_epoch_{epoch}.pth')
        
        print(f"\nSemi-supervised training completed! Best validation accuracy: {self.best_accuracy:.2f}%")
