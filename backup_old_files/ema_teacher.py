import torch
import torch.nn as nn
import copy
from typing import Dict, Any


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
            
        print(f"EMA Teacher initialized with momentum: {momentum}")
    
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
        print(f"Teacher model loaded from {path}")


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between student and teacher predictions.
    Uses MSE loss between softmax outputs.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 1.0):
        """
        Initialize consistency loss.
        
        Args:
            temperature: Temperature for softmax (higher = softer probabilities)
            alpha: Weight for consistency loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
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
        # Apply temperature scaling and softmax
        student_probs = torch.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute MSE loss between probability distributions
        consistency_loss = self.mse_loss(student_probs, teacher_probs)
        
        return self.alpha * consistency_loss


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss between student and teacher predictions.
    Alternative to MSE-based consistency loss.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 1.0):
        """
        Initialize KL divergence loss.
        
        Args:
            temperature: Temperature for softmax
            alpha: Weight for KL loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
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
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute KL divergence
        kl_loss = self.kl_div(student_log_probs, teacher_probs)
        
        # Scale by temperature squared (standard practice)
        return self.alpha * kl_loss * (self.temperature ** 2)
