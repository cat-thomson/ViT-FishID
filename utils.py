import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from typing import Dict, Any, Tuple, List


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        output: Model predictions [batch_size, num_classes]
        target: Ground truth labels [batch_size]
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(
    state: Dict[str, Any], 
    is_best: bool, 
    checkpoint_dir: str, 
    filename: str = 'checkpoint.pth'
) -> None:
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        shutil.copyfile(filepath, best_filepath)
        print(f"Best model saved to {best_filepath}")


def load_checkpoint(
    filepath: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['student_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze the backbone of a model.
    
    Args:
        model: Model to modify
        freeze: Whether to freeze (True) or unfreeze (False)
    """
    # Assuming the model has a 'backbone' attribute
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        print(f"Backbone parameters {status}")
    else:
        print("Model does not have a 'backbone' attribute")


def get_learning_rate_schedule(
    optimizer: torch.optim.Optimizer,
    schedule_type: str = 'cosine',
    epochs: int = 100,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        schedule_type: Type of schedule ('cosine', 'step', 'linear')
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    if schedule_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
        )
    elif schedule_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.1
        )
    elif schedule_type == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=min_lr, total_iters=epochs
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': all_size,
        'param_count': param_sum,
        'buffer_count': buffer_sum
    }


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should be stopped
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
