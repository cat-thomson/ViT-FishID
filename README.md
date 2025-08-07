# 🐟 ViT-FishID: Semi-Supervised Fish Classification with Vision Transformers

A comprehensive implementation of EMA (Exponential Moving Average) teacher-student framework for semi-supervised fish classification using Vision Transformers (ViT).

##  Overview

This project implements a state-of-the-art semi-supervised learning pipeline specifically designed for fish classification. It combines:

- **EMA Teacher-Student Framework**: Consistency regularization using exponential moving averages
- **Vision Transformers (ViT)**: Modern transformer architecture for image classification
- **Semi-Supervised Learning**: Efficiently use both labeled and unlabeled fish data
- **Automated Data Pipeline**: Extract fish cutouts from bounding box annotations
- **Interactive Dataset Organization**: Smart species detection and organization

## 🏗️ Project Structure

```
ViT-FishID/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                     # Training configuration
│
├── 🧠 Model & Training
├── vit_model.py                  # ViT architecture implementations
├── ema_teacher.py                # EMA teacher and consistency losses
├── ema_trainer.py                # Supervised EMA trainer
├── semi_supervised_trainer.py    # Semi-supervised trainer
├── data_loader.py                # Basic data loading utilities
├── semi_supervised_data.py       # Mixed labeled/unlabeled data loaders
│
├── 🎯 Training Scripts
├── main.py                       # Supervised training
├── main_semi_supervised.py       # Semi-supervised training
│
├── 📁 Data Pipeline
├── extract_fish_cutouts.py       # Extract fish from bounding boxes
├── organize_fish_data.py         # Organize extracted fish images
├── fish_pipeline.py              # Complete extraction + organization pipeline
└── example_species_mapping.json  # Example species ID mapping
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

If you have frame images with bounding box annotations (YOLO format):

```bash
# Run complete pipeline: extraction + organization
python fish_pipeline.py \
    --frames_dir /path/to/your/frames \
    --annotations_dir /path/to/your/annotations \
    --output_dir /path/to/organized/dataset \
    --species_mapping_file species_mapping.json
```

If you already have fish cutout images:

```bash
# Just organize existing cutouts
python organize_fish_data.py \
    --input_dir /path/to/fish/cutouts \
    --output_dir /path/to/organized/dataset
```

### 3. Train Your Model

#### Option A: Google Colab (Recommended for beginners)

1. **Upload to GitHub**: Commit your code to GitHub (images will be ignored via .gitignore)
2. **Upload Images**: Put your fish images in Google Drive
3. **Open Colab**: Use the provided `Colab_Training.ipynb` notebook
4. **Run Training**: Follow the notebook cells to train your model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cat-thomson/ViT-FishID/blob/main/Colab_Training.ipynb)

#### Option B: Local Training

##### Semi-Supervised Training (Recommended)
```bash
python main_semi_supervised.py \
    --data_dir /path/to/organized/dataset \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --use_wandb
```

##### Supervised Training (Labeled data only)
```bash
python main.py \
    --data_dir /path/to/organized/dataset \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4
```

## 🌐 Google Colab Training

For users who want to train without local GPU setup, we provide a complete Google Colab notebook.

### Why Use Google Colab?
- ✅ **Free GPU Access**: Train on Tesla T4 GPUs for free
- ✅ **No Setup Required**: Pre-configured environment
- ✅ **Easy Data Management**: Connect with Google Drive
- ✅ **Automatic Downloads**: Get your trained models easily

### Colab Setup Steps

1. **Prepare Your Data in Google Drive**
   ```
   /MyDrive/Fish_Images/
   ├── bass_001.jpg
   ├── trout_002.jpg
   ├── salmon_003.jpg
   └── ...
   ```

2. **Open the Colab Notebook**
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cat-thomson/ViT-FishID/blob/main/Colab_Training.ipynb)

3. **Enable GPU Runtime**
   - Runtime → Change runtime type → Hardware accelerator → GPU

4. **Update Paths in Notebook**
   ```python
   FISH_IMAGES_PATH = "/content/drive/MyDrive/Your_Fish_Folder"  # Update this!
   ```

5. **Run All Cells**
   - The notebook will handle everything automatically
   - Training progress will be shown in real-time
   - Results will be automatically downloaded

### Colab Training Features
- 🔄 **Automatic Data Organization**: Sorts labeled/unlabeled fish
- 📊 **Real-time Monitoring**: W&B integration for tracking
- � **Automatic Saving**: Checkpoints saved to Google Drive
- ⬇️ **Easy Download**: Results packaged and downloaded
- 🧹 **Smart Cleanup**: Optional space management

### Expected Colab Performance
- **Training Time**: ~2-3 hours for 50 epochs
- **GPU Memory**: ~6-8GB (fits comfortably on T4)
- **Storage**: ~2-5GB in Google Drive
- **Accuracy**: Same as local training (~75-85%)


### Data Pipeline

#### 1. Extract Fish Cutouts from Bounding Boxes

If your data consists of frame images with YOLO format annotations:

```bash
python extract_fish_cutouts.py \
    --frames_dir /path/to/frames \
    --annotations_dir /path/to/annotations \
    --output_dir /path/to/extracted/cutouts \
    --buffer_ratio 0.1 \
    --min_size 50
```

**Parameters:**
- `--frames_dir`: Directory containing frame images
- `--annotations_dir`: Directory containing YOLO format annotation files (.txt)
- `--output_dir`: Where to save extracted fish cutouts
- `--buffer_ratio`: Extra padding around bounding box (0.1 = 10% padding)
- `--min_size`: Minimum cutout size in pixels (filters out tiny detections)

#### 2. Organize Fish Images for Training

```bash
# Interactive mode (default)
python organize_fish_data.py \
    --input_dir /path/to/fish/cutouts \
    --output_dir /path/to/organized/dataset

# Non-interactive mode with predefined species
python organize_fish_data.py \
    --input_dir /path/to/fish/cutouts \
    --output_dir /path/to/organized/dataset \
    --labeled_species bass trout salmon \
    --no-interactive
```

This creates the following structure:
```
organized_dataset/
├── labeled/
│   ├── bass/       # Images containing "bass" in filename
│   ├── trout/      # Images containing "trout" in filename
│   └── salmon/     # Images containing "salmon" in filename
└── unlabeled/      # All other fish images
```

#### 3. Complete Pipeline (One Command)

```bash
python fish_pipeline.py \
    --frames_dir /path/to/frames \
    --annotations_dir /path/to/annotations \
    --output_dir /path/to/final/dataset \
    --buffer_ratio 0.1 \
    --min_cutout_size 50 \
    --species_mapping_file my_species.json \
    --labeled_species bass trout salmon
```

### Training Configuration

Edit `config.py` to customize training parameters:

```python
class Config:
    # Model settings
    model_name = 'vit_base_patch16_224'  # or 'microsoft/vit-base-patch16-224'
    num_classes = 10  # Number of fish species
    image_size = 224
    
    # Training settings
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100
    
    # EMA settings
    ema_momentum = 0.999
    consistency_weight = 1.0
    
    # Semi-supervised settings
    confidence_threshold = 0.95
    consistency_ramp_epochs = 20
```

### Advanced Features

#### Custom Species Mapping

Create a JSON file mapping species IDs to names:

```json
{
    "0": "unknown",
    "1": "bass",
    "2": "trout", 
    "3": "salmon",
    "4": "tuna"
}
```

#### Weights & Biases Integration

```bash
# Login to W&B
wandb login

# Train with logging
python main_semi_supervised.py \
    --data_dir /path/to/dataset \
    --use_wandb \
    --wandb_project "fish-classification" \
    --wandb_entity "your-username"
```

#### Resume Training

```bash
python main_semi_supervised.py \
    --data_dir /path/to/dataset \
    --resume_from /path/to/checkpoint.pth
```

## 🧠 Model Architecture

### Vision Transformer (ViT)
- **Base Model**: ViT-Base (86M parameters)
- **Input Size**: 224×224 RGB images
- **Patch Size**: 16×16 patches
- **Implementation**: Compatible with both `timm` and HuggingFace `transformers`

### EMA Teacher-Student Framework
- **Teacher Model**: Exponential moving average of student weights
- **Student Model**: Standard ViT trained with supervised loss
- **Consistency Loss**: MSE or KL-divergence between teacher and student predictions
- **EMA Momentum**: 0.999 (configurable)

### Semi-Supervised Learning
- **Labeled Data**: Standard cross-entropy loss
- **Unlabeled Data**: Consistency regularization between teacher and student
- **Pseudo-Labeling**: High-confidence teacher predictions used as labels
- **Confidence Threshold**: 0.95 (configurable)

## 📊 Training Monitoring

The training process logs comprehensive metrics:

### Supervised Metrics
- Training/Validation Loss
- Training/Validation Accuracy
- Learning Rate Schedule
- Model Checkpoints

### Semi-Supervised Metrics
- Consistency Loss (teacher vs student agreement)
- Pseudo-Label Accuracy (quality of teacher predictions)
- Labeled vs Unlabeled Loss Breakdown
- Teacher Model Performance

### Example Training Output
```
Epoch 1/100:
📚 Labeled batch - Loss: 2.456, Acc: 34.2%
🔄 Unlabeled batch - Consistency: 0.123, Pseudo-acc: 67.8%
📊 Validation - Loss: 2.234, Acc: 42.1%
💾 Checkpoint saved: best_model_epoch_1.pth
```

## 🎛️ Hyperparameter Tuning

### Key Parameters to Tune

1. **Learning Rate**: Start with 1e-4, try 5e-5 or 2e-4
2. **Consistency Weight**: Balance labeled vs consistency loss (0.1 - 2.0)
3. **EMA Momentum**: Teacher update rate (0.995 - 0.9999)
4. **Confidence Threshold**: Pseudo-label quality gate (0.9 - 0.98)
5. **Batch Size**: Depends on GPU memory (16, 32, 64)

### Recommended Starting Points
```python
# Conservative (stable but slower)
learning_rate = 5e-5
consistency_weight = 0.5
ema_momentum = 0.999

# Aggressive (faster but may be unstable)
learning_rate = 2e-4
consistency_weight = 2.0
ema_momentum = 0.995
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 16`
   - Use gradient accumulation in `config.py`

2. **Poor Consistency Loss**
   - Lower learning rate: `--learning_rate 5e-5`
   - Increase EMA momentum: `ema_momentum = 0.9999`

3. **Low Pseudo-Label Accuracy**
   - Increase confidence threshold: `confidence_threshold = 0.98`
   - Train longer on labeled data first

4. **Data Loading Errors**
   - Check file permissions
   - Verify image formats (JPG, PNG supported)
   - Ensure proper directory structure

### Performance Tips

1. **Use Mixed Precision**: Faster training with minimal accuracy loss
2. **Data Loading**: Increase `num_workers` in data loaders
3. **Augmentation**: Balance between diversity and label preservation
4. **Validation**: Use stratified sampling for balanced evaluation

## 📈 Expected Results

### Performance Benchmarks
- **Supervised Baseline**: ~75-85% accuracy (depends on dataset)
- **Semi-Supervised**: +5-15% improvement over supervised
- **Training Time**: ~2-4 hours on single GPU for 100 epochs

### Convergence Patterns
- **First 20 epochs**: Rapid improvement on labeled data
- **Epochs 20-60**: Consistency loss stabilizes, pseudo-labels improve
- **Epochs 60+**: Fine-tuning and final convergence

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional augmentation strategies
- Multi-scale training approaches
- Advanced pseudo-labeling techniques
- Performance optimizations

## 📚 References

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Mean Teacher](https://arxiv.org/abs/1703.01780)
- [FixMatch](https://arxiv.org/abs/2001.07685)
- [Semi-Supervised Learning Literature](https://github.com/yassouali/awesome-semi-supervised-learning)

## 📄 License

MIT License - feel free to use for research and commercial applications.

---

🐟 Happy fish classification! If you have questions or run into issues, please check the troubleshooting section or open an issue.
