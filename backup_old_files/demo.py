"""
Simple example demonstrating the EMA Teacher-Student concept.
This shows the core ideas without requiring the full dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple


# Simplified ViT-like model for demonstration
class SimpleViT(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


# Simple EMA Teacher
class SimpleEMATeacher:
    def __init__(self, student_model: nn.Module, momentum: float = 0.999):
        self.momentum = momentum
        self.teacher_model = self._create_teacher_copy(student_model)
        
    def _create_teacher_copy(self, student_model: nn.Module) -> nn.Module:
        """Create teacher as a copy of student."""
        teacher = type(student_model)(num_classes=10)  # Assuming 10 classes
        teacher.load_state_dict(student_model.state_dict())
        teacher.eval()
        
        # Disable gradients for teacher
        for param in teacher.parameters():
            param.requires_grad = False
            
        return teacher
    
    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """Update teacher using EMA."""
        for teacher_param, student_param in zip(
            self.teacher_model.parameters(),
            student_model.parameters()
        ):
            teacher_param.data.mul_(self.momentum).add_(
                student_param.data, alpha=1 - self.momentum
            )


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    """Compute consistency loss between student and teacher."""
    student_probs = torch.softmax(student_logits / temperature, dim=1)
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
    
    return nn.MSELoss()(student_probs, teacher_probs)


def create_dummy_data(batch_size: int = 8, num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy data for demonstration."""
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


def demonstrate_ema_training():
    """Demonstrate EMA teacher-student training."""
    print("🐟 EMA Teacher-Student Training Demonstration")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create models
    student_model = SimpleViT(num_classes=10).to(device)
    ema_teacher = SimpleEMATeacher(student_model, momentum=0.999)
    
    # Setup training
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=0.05)
    supervised_loss_fn = nn.CrossEntropyLoss()
    
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    print(f"EMA momentum: {ema_teacher.momentum}")
    
    # Training loop demonstration
    student_model.train()
    ema_teacher.teacher_model.eval()
    
    print("\nTraining Steps:")
    print("-" * 30)
    
    for step in range(10):  # 10 demonstration steps
        # Generate dummy batch
        images, labels = create_dummy_data(batch_size=8)
        images, labels = images.to(device), labels.to(device)
        
        # Student forward pass
        student_logits = student_model(images)
        sup_loss = supervised_loss_fn(student_logits, labels)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_logits = ema_teacher.teacher_model(images)
        
        # Consistency loss
        cons_loss = consistency_loss(student_logits, teacher_logits)
        
        # Total loss
        total_loss = sup_loss + cons_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update EMA teacher
        ema_teacher.update(student_model)
        
        # Print progress
        student_acc = (student_logits.argmax(dim=1) == labels).float().mean() * 100
        print(f"Step {step+1:2d}: Loss={total_loss:.4f} (Sup={sup_loss:.4f}, Cons={cons_loss:.4f}), Acc={student_acc:.1f}%")
    
    print("\n✅ EMA Training Demonstration Complete!")
    
    # Show the difference between student and teacher
    print("\nModel Comparison:")
    print("-" * 20)
    
    with torch.no_grad():
        test_images, test_labels = create_dummy_data(batch_size=4)
        test_images = test_images.to(device)
        
        student_model.eval()
        student_out = student_model(test_images)
        teacher_out = ema_teacher.teacher_model(test_images)
        
        student_probs = torch.softmax(student_out, dim=1)
        teacher_probs = torch.softmax(teacher_out, dim=1)
        
        print(f"Student max prob: {student_probs.max(dim=1)[0].mean():.3f}")
        print(f"Teacher max prob: {teacher_probs.max(dim=1)[0].mean():.3f}")
        print(f"Prediction similarity: {(student_out.argmax(dim=1) == teacher_out.argmax(dim=1)).float().mean():.3f}")


def explain_ema_benefits():
    """Explain the benefits of EMA teacher-student training."""
    print("\n🎯 Benefits of EMA Teacher-Student Training")
    print("=" * 45)
    
    benefits = [
        "🔄 **Stability**: Teacher provides stable targets, reducing training noise",
        "📈 **Better Convergence**: Exponential averaging smooths parameter updates",
        "🎯 **Improved Accuracy**: Teacher often outperforms student by 1-3%",
        "🛡️ **Regularization**: Consistency loss acts as implicit regularization",
        "⚖️ **Balanced Learning**: Combines supervised and self-supervised signals",
        "🔧 **Easy Integration**: Drop-in replacement for standard training"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print(f"\n💡 **Key Insight**: The teacher model accumulates knowledge over time,")
    print(f"   providing more stable and reliable pseudo-labels than the student.")


def show_hyperparameter_guide():
    """Show hyperparameter guidance."""
    print("\n⚙️ Hyperparameter Guide")
    print("=" * 25)
    
    params = {
        "EMA Momentum": {
            "range": "0.99 - 0.9999",
            "default": "0.999",
            "note": "Higher = more stable teacher, slower adaptation"
        },
        "Consistency Weight": {
            "range": "0.1 - 10.0",
            "default": "1.0",
            "note": "Balance between supervised and consistency loss"
        },
        "Temperature": {
            "range": "1.0 - 8.0",
            "default": "4.0",
            "note": "Higher = softer probability distributions"
        },
        "Learning Rate": {
            "range": "1e-5 - 1e-3",
            "default": "1e-4",
            "note": "Often slightly lower than standard training"
        }
    }
    
    for param, info in params.items():
        print(f"\n{param}:")
        print(f"  Range: {info['range']}")
        print(f"  Default: {info['default']}")
        print(f"  Note: {info['note']}")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_ema_training()
    
    # Show explanations
    explain_ema_benefits()
    show_hyperparameter_guide()
    
    print(f"\n🚀 Ready to train your ViT-FishID model!")
    print(f"   Use: python main.py --data_dir /path/to/fish/dataset --use_wandb")
