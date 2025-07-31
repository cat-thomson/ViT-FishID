# ViT-FishID: Vision Transformer with EMA Teacher-Student Framework

A comprehensive implementation of Vision Transformer (ViT-Base) using an Exponential Moving Average (EMA) teacher-student framework for fish species classification, with **full semi-supervised learning support**.

## 🐟 Overview

This project implements a state-of-the-art training framework that combines:

- **Vision Transformer (ViT-Base)**: Pre-trained transformer architecture for image classification
- **EMA Teacher-Student Framework**: Exponential moving average updates for improved training stability
- **Semi-Supervised Learning**: Leverage both labeled and unlabeled fish images
- **Advanced Data Augmentation**: Strong and weak augmentation strategies
- **Comprehensive Logging**: Integration with Weights & Biases

### Key Features

- ✅ **Semi-Supervised Learning**: Use unlabeled fish cutouts to improve performance
- ✅ **Dual Model Architecture**: Student model learns with gradients, teacher provides stable predictions
- ✅ **Consistency Regularization**: MSE or KL divergence loss between student and teacher
- ✅ **Intelligent Data Organization**: Automatically organize fish cutouts by species
- ✅ **Advanced Augmentation**: Albumentations-based augmentation pipeline
- ✅ **Flexible Backbone**: Support for both timm and Hugging Face transformers
- ✅ **Production Ready**: Comprehensive logging, checkpointing, and resuming

## 🏗️ Architecture

### EMA Teacher-Student Framework

```
Input Image
     │
     ├── Strong Augmentation ──→ Student Model ──→ Supervised Loss
     │                              │
     └── Weak Augmentation ────→ Teacher Model ──→ Consistency Loss
                                     ↑
                               EMA Update
```

### Key Components

1. **Student Model**: ViT-Base trained with standard backpropagation
2. **Teacher Model**: EMA copy of student providing stable pseudo-labels
3. **Consistency Loss**: Encourages agreement between student and teacher
4. **EMA Updates**: `teacher = momentum * teacher + (1 - momentum) * student`

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/ViT-FishID.git
cd ViT-FishID
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv vit_env
source vit_env/bin/activate  # On Windows: vit_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Additional Packages

```bash
# For GPU support (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (MPS)
pip install torch torchvision
```

## 📊 Dataset Preparation

### Expected Directory Structure

```
fish_dataset/
├── species_1/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── species_2/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── species_n/
    ├── img_001.jpg
    └── ...
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 🚀 Quick Start

### For Semi-Supervised Learning (Recommended for Mixed Labeled/Unlabeled Data)

If you have fish cutouts but not all are labeled by species:

#### 1. Organize Your Data
```bash
# Automatically organize fish cutouts
python organize_fish_data.py \
    --input_dir /path/to/fish/cutouts \
    --output_dir /path/to/organized/dataset \
    --interactive
```

#### 2. Train with Semi-Supervised Learning
```bash
python main_semi_supervised.py \
    --data_dir /path/to/organized/dataset \
    --epochs 100 \
    --unlabeled_ratio 3.0 \
    --use_wandb
```

### For Fully Supervised Learning

If all your fish images are labeled:

### For Fully Supervised Learning

If all your fish images are labeled:

#### Basic Training

```bash
python main.py \
    --data_dir /path/to/fish/dataset \
    --epochs 100 \
    --batch_size 32 \
    --use_wandb
```

#### Advanced Training with Custom Parameters

```bash
python main.py \
    --data_dir /path/to/fish/dataset \
    --model_name vit_base_patch16_224 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --ema_momentum 0.999 \
    --consistency_weight 1.0 \
    --consistency_loss mse \
    --temperature 4.0 \
    --warmup_epochs 10 \
    --use_wandb \
    --wandb_project my-fish-project
```

#### Using the Training Script

```bash
# Edit train.sh to set your data path
nano train.sh

# Run training
./train.sh
```

## ⚙️ Configuration

### Key Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `ema_momentum` | EMA momentum for teacher updates | 0.999 | 0.9-0.9999 |
| `consistency_weight` | Weight for consistency loss | 1.0 | 0.1-10.0 |
| `temperature` | Temperature for softmax | 4.0 | 1.0-8.0 |
| `learning_rate` | Learning rate for optimizer | 1e-4 | 1e-5-1e-3 |
| `warmup_epochs` | Number of warmup epochs | 10 | 5-20 |

### EMA Teacher-Student Specific Settings

```python
# EMA momentum (higher = more stable teacher)
ema_momentum = 0.999  # or 0.9999 for very stable teacher

# Consistency loss type
consistency_loss = "mse"  # or "kl" for KL divergence

# Consistency weight (balance supervised vs consistency loss)
consistency_weight = 1.0  # increase for stronger regularization
```

## 📈 Monitoring and Logging

### Weights & Biases Integration

```bash
# Login to wandb (first time only)
wandb login

# Train with logging
python main.py --use_wandb --wandb_project my-project
```

### Metrics Tracked

- **Training**: Supervised loss, consistency loss, accuracy
- **Validation**: Student and teacher performance
- **System**: Learning rate, memory usage, training time

## 🔄 Model Architecture Details

### Vision Transformer Configuration

```python
# Default ViT-Base configuration
{
    "model_name": "vit_base_patch16_224",
    "image_size": 224,
    "patch_size": 16,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "intermediate_size": 3072
}
```

### EMA Update Formula

```python
# Teacher parameter update
teacher_param = ema_momentum * teacher_param + (1 - ema_momentum) * student_param
```

### Consistency Loss Functions

**MSE Loss:**
```python
student_probs = softmax(student_logits / temperature)
teacher_probs = softmax(teacher_logits / temperature)
loss = MSE(student_probs, teacher_probs)
```

**KL Divergence:**
```python
student_log_probs = log_softmax(student_logits / temperature)
teacher_probs = softmax(teacher_logits / temperature)
loss = KL_div(student_log_probs, teacher_probs) * temperature²
```

## 🎯 Performance Tips

### For Better Results

1. **Higher EMA Momentum**: Use 0.9999 for very stable teacher
2. **Temperature Tuning**: Higher temperature (6-8) for softer targets
3. **Consistency Weight**: Start with 1.0, increase gradually
4. **Warmup**: Essential for stable training start

### Memory Optimization

```bash
# Reduce batch size for limited GPU memory
--batch_size 16

# Use gradient checkpointing (add to model)
--gradient_checkpointing

# Mixed precision training (requires apex)
--use_amp
```

## 📋 Example Results

### Training Progress

```
Epoch 50/100
Train - Loss: 0.234, Acc: 92.5%
Student Val - Loss: 0.198, Acc: 94.1%
Teacher Val - Loss: 0.186, Acc: 95.2%
```

### Expected Performance

| Dataset Size | Epochs | Student Acc | Teacher Acc |
|--------------|--------|-------------|-------------|
| 1K images | 100 | 85-90% | 87-92% |
| 5K images | 100 | 90-95% | 92-96% |
| 10K+ images | 100 | 95-98% | 96-99% |

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce batch size
--batch_size 16

# Reduce image size
--image_size 192
```

**Poor Convergence:**
```bash
# Increase warmup
--warmup_epochs 20

# Reduce learning rate
--learning_rate 5e-5
```

**Teacher-Student Divergence:**
```bash
# Lower EMA momentum
--ema_momentum 0.99

# Reduce consistency weight
--consistency_weight 0.5
```

## 📚 References

1. **Vision Transformer**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
2. **EMA Teacher-Student**: "Mean teachers are better role models" (Tarvainen & Valpola, 2017)
3. **Consistency Regularization**: "Temporal Ensembling for Semi-Supervised Learning" (Laine & Aila, 2016)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for ViT implementation
- timm library for vision models
- Albumentations for data augmentation
- Weights & Biases for experiment tracking

---

**Happy Training! 🐟🤖**
