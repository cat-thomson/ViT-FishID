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
from data_loader import create_fish_dataloaders
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint


class EMATrainer:
    """
    Trainer for EMA Teacher-Student framework with ViT.
    """
    
    def __init__(
        self,
        student_model: ViTForFishClassification,
        num_classes: int,
        device: str = 'cuda',
        ema_momentum: float = 0.999,
        consistency_loss_type: str = 'mse',
        consistency_weight: float = 1.0,
        temperature: float = 4.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        use_wandb: bool = True
    ):
        """
        Initialize EMA Trainer.
        
        Args:
            student_model: Student ViT model
            num_classes: Number of classes
            device: Training device
            ema_momentum: EMA momentum for teacher updates
            consistency_loss_type: 'mse' or 'kl' for consistency loss
            consistency_weight: Weight for consistency loss
            temperature: Temperature for softmax in consistency loss
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            use_wandb: Whether to use Weights & Biases logging
        """
        self.device = device
        self.num_classes = num_classes
        self.consistency_weight = consistency_weight
        self.warmup_epochs = warmup_epochs
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
        
        # Learning rate scheduler (cosine annealing with warmup)
        self.scheduler = None  # Will be set in train method
        
        # Metrics tracking
        self.best_accuracy = 0.0
        self.current_epoch = 0
        
        print(f"EMA Trainer initialized:")
        print(f"  - EMA momentum: {ema_momentum}")
        print(f"  - Consistency loss: {consistency_loss_type}")
        print(f"  - Consistency weight: {consistency_weight}")
        print(f"  - Temperature: {temperature}")
    
    def _warmup_learning_rate(self, epoch: int, batch_idx: int, total_batches: int):
        """Apply learning rate warmup."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_total_iters = self.warmup_epochs * total_batches
            current_iter = epoch * total_batches + batch_idx
            lr_scale = min(1.0, current_iter / warmup_total_iters)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
    
    def train_epoch(
        self, 
        labeled_loader: DataLoader, 
        unlabeled_loader: Optional[DataLoader] = None,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            labeled_loader: DataLoader for labeled data
            unlabeled_loader: DataLoader for unlabeled data (for semi-supervised learning)
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.student_model.train()
        self.ema_teacher.teacher_model.eval()
        
        # Metrics
        supervised_losses = AverageMeter()
        consistency_losses = AverageMeter()
        total_losses = AverageMeter()
        top1_acc = AverageMeter()
        
        # Progress bar
        pbar = tqdm(labeled_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)
            
            # Apply warmup
            self._warmup_learning_rate(epoch, batch_idx, len(labeled_loader))
            
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
            
            # Gradient clipping (optional but recommended for ViT)
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
            for images, targets in tqdm(val_loader, desc='Validating'):
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_dir: str = './checkpoints',
        save_frequency: int = 10
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N epochs
        """
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        
        print(f"Starting training for {epochs} epochs...")
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
            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Student Val - Loss: {student_val_metrics['loss']:.4f}, Acc: {student_val_metrics['top1_accuracy']:.2f}%")
            print(f"Teacher Val - Loss: {teacher_val_metrics['loss']:.4f}, Acc: {teacher_val_metrics['top1_accuracy']:.2f}%")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['total_loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
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
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_accuracy:.2f}%")
