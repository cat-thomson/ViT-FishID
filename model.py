"""
Unified Model Module for ViT-FishID.

This module contains:
- ViTForFishClassification: Vision Transformer model for fish classification
- EMATeacher: Exponential Moving Average teacher for semi-supervised learning
- ConsistencyLoss: Loss functions for teacher-student training

Author: GitHub Copilot
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import timm
from transformers import ViTConfig, ViTModel
from typing import Optional, Tuple, Dict, Any


class ViTForFishClassification(nn.Module):
    """
    Vision Transformer for Fish Classification.
    Uses timm implementation for better performance.
    """
    
    def __init__(
        self, 
        num_classes: int,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize ViT model for fish classification.
        
        Args:
            num_classes: Number of fish species classes
            model_name: Model architecture name (default: vit_base_patch16_224)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for classification head
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Use timm implementation (recommended for vision tasks)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='token'  # Use CLS token pooling
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Initialize classification head
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the classification head weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Features [batch_size, feature_dim]
        """
        return self.backbone(x)


class EMATeacher:
    """
    Exponential Moving Average Teacher for teacher-student training framework.
    
    The teacher model is updated as an exponential moving average of the student model:
    teacher_param = momentum * teacher_param + (1 - momentum) * student_param
    """
    
    def __init__(self, student_model: nn.Module, momentum: float = 0.999, device: str = 'cuda'):
        """
        Initialize EMA Teacher.
        
        Args:
            student_model: The student model to create teacher from
            momentum: EMA momentum coefficient (typically 0.999 or 0.9999)
            device: Device to store teacher model
        """
        self.momentum = momentum
        self.device = device
        
        # Create teacher as a deep copy of student
        self.teacher_model = copy.deepcopy(student_model)
        self.teacher_model.to(device)
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Disable gradients for teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        print(f"✅ EMA Teacher initialized with momentum: {momentum}")
    
    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """
        Update teacher parameters using EMA.
        
        Args:
            student_model: Current student model to update teacher from
        """
        for teacher_param, student_param in zip(
            self.teacher_model.parameters(), 
            student_model.parameters()
        ):
            teacher_param.data.mul_(self.momentum).add_(
                student_param.data, alpha=1 - self.momentum
            )
    
    def get_teacher_model(self) -> nn.Module:
        """Get the teacher model."""
        return self.teacher_model
    
    def save_teacher_state(self, path: str):
        """Save teacher model state."""
        torch.save({
            'teacher_state_dict': self.teacher_model.state_dict(),
            'momentum': self.momentum
        }, path)
    
    def load_teacher_state(self, path: str):
        """Load teacher model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        self.momentum = checkpoint['momentum']
        print(f"✅ Teacher model loaded from {path}")


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between student and teacher predictions.
    Uses MSE loss between softmax outputs with temperature scaling.
    """
    
    def __init__(self, temperature: float = 4.0):
        """
        Initialize consistency loss.
        
        Args:
            temperature: Temperature for softmax (higher = softer probabilities)
        """
        super().__init__()
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss between student and teacher.
        
        Args:
            student_logits: Logits from student model [batch_size, num_classes]
            teacher_logits: Logits from teacher model [batch_size, num_classes]
            
        Returns:
            Consistency loss value
        """
        # Ensure both tensors are on the same device and have the same shape
        assert student_logits.shape == teacher_logits.shape, f"Shape mismatch: {student_logits.shape} vs {teacher_logits.shape}"
        
        # Apply temperature scaling and softmax
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # MSE loss between probability distributions
        loss = self.mse_loss(student_probs, teacher_probs)
        
        # Scale by temperature squared to maintain gradient magnitude
        loss = loss * (self.temperature ** 2)
        
        return loss


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss between student and teacher predictions.
    Alternative to MSE consistency loss.
    """
    
    def __init__(self, temperature: float = 4.0):
        """
        Initialize KL divergence loss.
        
        Args:
            temperature: Temperature for softmax
        """
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between student and teacher.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            
        Returns:
            KL divergence loss value
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence loss
        loss = self.kl_loss(student_log_probs, teacher_probs)
        
        return loss * (self.temperature ** 2)


def create_model_and_teacher(
    num_classes: int, 
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    dropout_rate: float = 0.1,
    ema_momentum: float = 0.999,
    device: str = 'cuda'
) -> Tuple[ViTForFishClassification, EMATeacher]:
    """
    Create student model and EMA teacher.
    
    Args:
        num_classes: Number of fish species classes
        model_name: ViT model architecture
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for classification head
        ema_momentum: EMA momentum for teacher updates
        device: Device to place models on
        
    Returns:
        Tuple of (student_model, ema_teacher)
    """
    # Create student model
    student_model = ViTForFishClassification(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    student_model.to(device)
    
    # Create EMA teacher
    ema_teacher = EMATeacher(
        student_model=student_model,
        momentum=ema_momentum,
        device=device
    )
    
    return student_model, ema_teacher
