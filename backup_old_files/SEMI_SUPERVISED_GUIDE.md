# 🐟 Semi-Supervised Fish Classification Guide

## Quick Start for Unlabeled Fish Cutouts

If you have fish cutout images but not all are labeled by species, this guide will help you set up semi-supervised learning with the EMA teacher-student framework.

## 📂 Step 1: Organize Your Fish Cutouts

### Option A: Automatic Organization (Recommended)

Use the interactive organizer script:

```bash
python organize_fish_data.py \
    --input_dir /path/to/your/fish/cutouts \
    --output_dir /path/to/organized/dataset \
    --interactive
```

The script will:
1. 🔍 **Analyze filenames** to find potential species names
2. 🎯 **Interactive selection** - choose which species to label
3. 📁 **Auto-organize** images into `labeled/` and `unlabeled/` directories

### Option B: Manual Organization

Create this directory structure:

```
organized_dataset/
├── labeled/
│   ├── salmon/
│   │   ├── salmon_001.jpg
│   │   ├── salmon_002.jpg
│   │   └── ...
│   ├── trout/
│   │   ├── trout_001.jpg
│   │   └── ...
│   └── bass/
│       └── ...
└── unlabeled/
    ├── fish_cutout_001.jpg
    ├── fish_cutout_002.jpg
    ├── unknown_fish_001.jpg
    └── ...
```

## 🚀 Step 2: Train with Semi-Supervised Learning

### Basic Training

```bash
python main_semi_supervised.py \
    --data_dir /path/to/organized/dataset \
    --epochs 100 \
    --use_wandb
```

### Advanced Training

```bash
python main_semi_supervised.py \
    --data_dir /path/to/organized/dataset \
    --model_name vit_base_patch16_224 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --ema_momentum 0.999 \
    --consistency_weight 2.0 \
    --pseudo_label_threshold 0.95 \
    --unlabeled_ratio 3.0 \
    --ramp_up_epochs 20 \
    --use_wandb \
    --wandb_project my-fish-project
```

## ⚙️ Key Semi-Supervised Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `ema_momentum` | Teacher stability (higher = more stable) | 0.999 |
| `consistency_weight` | Strength of unlabeled data influence | 1.0 - 3.0 |
| `pseudo_label_threshold` | Confidence threshold for pseudo-labels | 0.95 |
| `unlabeled_ratio` | Unlabeled samples per labeled sample | 2.0 - 5.0 |
| `ramp_up_epochs` | Epochs to gradually increase consistency weight | 20 |
| `temperature` | Softness of probability distributions | 4.0 |

## 📊 How Semi-Supervised Learning Works

### Training Process

```
Input Batch (Mixed: Labeled + Unlabeled)
     │
     ├── Labeled Images ──────→ Student ──→ Supervised Loss (CrossEntropy)
     │                             │
     └── Unlabeled Images ────→ Student ──→ Consistency Loss with Teacher
                                   ↓
                              Teacher (EMA of Student)
                                   ↑
                              EMA Update (momentum=0.999)
```

### Benefits for Your Use Case

1. **🎯 Leverage All Data**: Use both labeled and unlabeled fish images
2. **📈 Better Accuracy**: Typically 2-5% improvement over supervised-only
3. **🔄 Self-Improving**: Teacher provides increasingly better pseudo-labels
4. **💪 Robustness**: Consistency regularization improves generalization

## 📈 Expected Performance

| Labeled Samples | Unlabeled Samples | Expected Accuracy Gain |
|----------------|-------------------|----------------------|
| 100 per class | 500+ total | +3-5% |
| 200 per class | 1000+ total | +2-4% |
| 500+ per class | 2000+ total | +1-3% |

## 🔧 Troubleshooting Common Issues

### Low Consistency Loss

**Problem**: Consistency loss stays near zero
**Solution**: 
- Increase `consistency_weight` to 2.0-3.0
- Lower `pseudo_label_threshold` to 0.9
- Increase `unlabeled_ratio`

### Teacher-Student Divergence

**Problem**: Teacher and student performance differ significantly
**Solution**:
- Lower `ema_momentum` to 0.99
- Increase `ramp_up_epochs` to 30
- Reduce `learning_rate` to 5e-5

### Poor Pseudo-Label Quality

**Problem**: Low confidence in teacher predictions
**Solution**:
- Increase `temperature` to 6.0-8.0
- Train longer with more labeled data first
- Use stronger data augmentation

## 💡 Pro Tips

### 1. Start with Quality Labels
- Ensure your labeled species are correctly identified
- Use clear, high-quality images for labeled data
- Balance labeled classes (similar number of images per species)

### 2. Optimize Unlabeled Data
- Include diverse fish poses and lighting conditions
- Remove very blurry or corrupted images
- Use 2-5x more unlabeled than labeled images

### 3. Monitor Training
- Watch consistency loss - should gradually decrease
- Check pseudo-label confidence - should improve over time
- Teacher accuracy should exceed student accuracy

### 4. Hyperparameter Tuning
- Start with default values
- Increase `consistency_weight` if unlabeled data is high quality
- Adjust `unlabeled_ratio` based on your dataset size

## 🎯 Real-World Example

Let's say you have:
- 200 labeled salmon images
- 150 labeled trout images
- 100 labeled bass images
- 2000 unlabeled fish cutouts

**Recommended setup**:
```bash
python main_semi_supervised.py \
    --data_dir /path/to/organized/dataset \
    --unlabeled_ratio 4.0 \
    --consistency_weight 2.0 \
    --ema_momentum 0.999 \
    --ramp_up_epochs 25 \
    --epochs 150
```

This will use ~1800 unlabeled images per epoch (450 labeled × 4.0 ratio) and gradually increase the consistency loss weight over 25 epochs.

## 📚 Further Reading

- [Mean teachers are better role models](https://arxiv.org/abs/1703.01780) - Original EMA teacher-student paper
- [FixMatch](https://arxiv.org/abs/2001.07685) - Advanced semi-supervised learning
- [Temporal Ensembling](https://arxiv.org/abs/1610.02242) - Consistency regularization foundations
