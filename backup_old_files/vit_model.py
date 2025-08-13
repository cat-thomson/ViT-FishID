import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
import timm
from typing import Optional, Tuple


class ViTForFishClassification(nn.Module):
    """
    Vision Transformer for Fish Classification.
    Can use either Hugging Face transformers or timm implementation.
    """
    
    def __init__(
        self, 
        num_classes: int,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_timm: bool = True
    ):
        """
        Initialize ViT model for fish classification.
        
        Args:
            num_classes: Number of fish species classes
            model_name: Model architecture name
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for classification head
            use_timm: Whether to use timm library (recommended) or transformers
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_timm = use_timm
        
        if use_timm:
            # Use timm implementation (recommended for vision tasks)
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool='token'  # Use CLS token pooling
            )
            
            # Get feature dimension
            self.feature_dim = self.backbone.num_features
            
        else:
            # Use Hugging Face transformers implementation
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
            self.backbone = ViTModel.from_pretrained(
                'google/vit-base-patch16-224' if pretrained else None,
                config=config
            )
            self.feature_dim = config.hidden_size
        
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
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            return_features: Whether to return features before classification
            
        Returns:
            Logits [batch_size, num_classes] or features if return_features=True
        """
        if self.use_timm:
            # timm implementation
            features = self.backbone(x)  # [batch_size, feature_dim]
        else:
            # Hugging Face implementation
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0]  # CLS token
        
        if return_features:
            return features
            
        logits = self.classifier(features)
        return logits
    
    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor (backbone + projection)."""
        return nn.Sequential(
            self.backbone,
            self.classifier[:-1]  # Everything except final linear layer
        )


class ViTWithAugmentation(nn.Module):
    """
    ViT model with built-in augmentation for teacher-student training.
    Applies different augmentations to generate student and teacher inputs.
    """
    
    def __init__(
        self,
        base_model: ViTForFishClassification,
        strong_aug: Optional[nn.Module] = None,
        weak_aug: Optional[nn.Module] = None
    ):
        """
        Initialize ViT with augmentation.
        
        Args:
            base_model: Base ViT model
            strong_aug: Strong augmentation for student
            weak_aug: Weak augmentation for teacher
        """
        super().__init__()
        self.base_model = base_model
        self.strong_aug = strong_aug
        self.weak_aug = weak_aug
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'both',
        return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with augmentation.
        
        Args:
            x: Input images
            mode: 'student', 'teacher', or 'both'
            return_features: Whether to return features
            
        Returns:
            Logits (and features if requested)
        """
        if mode == 'student':
            if self.strong_aug is not None:
                x = self.strong_aug(x)
            return self.base_model(x, return_features=return_features)
        
        elif mode == 'teacher':
            if self.weak_aug is not None:
                x = self.weak_aug(x)
            return self.base_model(x, return_features=return_features)
        
        elif mode == 'both':
            # Apply augmentations and get both outputs
            student_x = self.strong_aug(x) if self.strong_aug is not None else x
            teacher_x = self.weak_aug(x) if self.weak_aug is not None else x
            
            student_output = self.base_model(student_x, return_features=return_features)
            teacher_output = self.base_model(teacher_x, return_features=return_features)
            
            return student_output, teacher_output
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'student', 'teacher', or 'both'")


def create_vit_model(num_classes: int, **kwargs) -> ViTForFishClassification:
    """
    Factory function to create ViT model.
    
    Args:
        num_classes: Number of fish species classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        ViT model for fish classification
    """
    return ViTForFishClassification(num_classes=num_classes, **kwargs)
