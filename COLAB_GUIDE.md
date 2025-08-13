# 🚀 Google Colab Setup Guide for ViT-FishID

This guide will walk you through running ViT-FishID semi-supervised training in Google Colab with GPU acceleration.

## 📋 Prerequisites

1. **Google Account**: You need a Google account to use Google Colab
2. **Fish Dataset**: Your organized fish dataset (with `labeled/` and `unlabeled/` folders)
3. **Google Drive**: Space to store your dataset and trained models

## 🏃‍♂️ Quick Start (5 Steps)

### Step 1: Open the Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "GitHub" tab
3. Enter: `cat-thomson/ViT-FishID`
4. Select: `ViT_FishID_Colab_Training.ipynb`
5. Click "Open in Colab"

### Step 2: Enable GPU Runtime
1. Click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: GPU
3. Click **Save**
4. Your runtime will restart

### Step 3: Upload Your Data to Google Drive
1. Open [Google Drive](https://drive.google.com/)
2. Upload your `fish_cutouts.zip` file to Google Drive
   - You can upload to the root directory or any folder
   - The notebook will automatically extract it for you
3. Note the path where you uploaded it (e.g., `/content/drive/MyDrive/fish_cutouts.zip`)

**Data Structure Inside ZIP:**
Your `fish_cutouts.zip` should contain:
   ```
   fish_cutouts/
   ├── labeled/
   │   ├── species_1/
   │   │   ├── fish_001.jpg
   │   │   └── fish_002.jpg
   │   └── species_2/
   │       └── ...
   └── unlabeled/
       ├── fish_003.jpg
       └── fish_004.jpg
   ```

### Step 4: Run the Notebook
1. **Execute each cell in order** (Shift + Enter)
2. **Mount Google Drive** when prompted (authorize access)
3. **Update the ZIP file path** in Step 5 to point to your `fish_cutouts.zip`
4. **Wait for automatic extraction** - the notebook will unzip your data
5. **Configure training parameters** in Step 7 (optional)
6. **Start training** in Step 8

### Step 5: Monitor Training
- Training will take 2-3 hours for 50 epochs
- Watch the progress and accuracy metrics
- Checkpoints are saved every 10 epochs

## 📊 Expected Performance

| Hardware | Batch Size | Training Time | Expected Accuracy |
|----------|------------|---------------|-------------------|
| Tesla T4 | 16 | 2-3 hours (50 epochs) | 75-80% |
| Tesla K80 | 8 | 4-5 hours (50 epochs) | 70-75% |
| Tesla V100 | 32 | 1-2 hours (50 epochs) | 80-85% |

## 🔧 Configuration Options

### Basic Settings (Recommended for Beginners)
```python
TRAINING_CONFIG = {
    'mode': 'semi_supervised',
    'epochs': 50,
    'batch_size': 16,  # Adjust based on GPU memory
    'val_split': 0.2,
    'test_split': 0.2,
}
```

### Advanced Settings (For Experienced Users)
```python
TRAINING_CONFIG = {
    'mode': 'semi_supervised',
    'epochs': 75,
    'batch_size': 24,
    'consistency_weight': 2.0,
    'pseudo_label_threshold': 0.7,
    'temperature': 4.0,
    'unlabeled_ratio': 2.0,
    'learning_rate': 1e-4,
    'use_wandb': True,  # Enable experiment tracking
}
```

## 🛠️ Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size
```python
TRAINING_CONFIG['batch_size'] = 8  # or even 4
```

### Problem: "Data directory not found"
**Solutions**:
1. Check the `ZIP_FILE_PATH` in Step 5 points to your `fish_cutouts.zip`
2. Ensure the ZIP file is uploaded to Google Drive
3. Verify the ZIP file name is exactly `fish_cutouts.zip`
4. Check that the ZIP file contains `labeled/` and `unlabeled/` folders

### Problem: "Runtime disconnected"
**Solutions**:
1. Colab sessions timeout after 12 hours
2. Keep the browser tab active
3. Training resumes from last checkpoint

### Problem: "Low accuracy (<50%)"
**Solutions**:
1. Increase training epochs to 75-100
2. Lower `pseudo_label_threshold` to 0.5
3. Check data quality and class balance

### Problem: "Consistency loss is 0.0000"
**Solutions**:
1. Lower `pseudo_label_threshold` to 0.5 or 0.6
2. Ensure you have unlabeled data
3. Verify `mode` is set to 'semi_supervised'

## 💡 Pro Tips

### 1. Data Upload Optimization
- **Use ZIP files**: Upload `fish_cutouts.zip` instead of individual folders for faster transfer
- **Check file size**: Large ZIP files (>2GB) may take longer to upload
- **Verify structure**: Ensure ZIP contains `labeled/` and `unlabeled/` folders
- **Check extraction**: The notebook will automatically extract and validate your data

### 2. Training Optimization
- **Start with fewer epochs** (25-50) to test setup
- **Use smaller batch sizes** if you get memory errors
- **Enable W&B logging** for better experiment tracking

### 3. Resource Management
- **Runtime management**: Keep Colab active to prevent timeout
- **Storage limits**: Free Google Drive has 15GB limit
- **GPU limits**: Free tier has daily GPU usage limits

### 4. Best Practices
- **Save frequently**: Models are auto-saved every 10 epochs
- **Download checkpoints**: Copy important models to local storage
- **Monitor progress**: Check validation accuracy trends
- **Experiment**: Try different hyperparameters

## 📥 After Training

### Download Your Model
After training completes, your model will be automatically saved to Google Drive:
```
/content/drive/MyDrive/ViT-FishID_Training_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── model_best.pth          # Best performing model
│   ├── model_latest.pth        # Most recent checkpoint
│   └── model_epoch_XX.pth      # Periodic checkpoints
├── training_config.json        # Your training settings
└── training_summary.txt        # Training summary
```

### Use Your Trained Model
```python
import torch
from model import ViTForFishClassification

# Load your trained model
model = ViTForFishClassification(num_classes=YOUR_NUM_CLASSES)
checkpoint = torch.load('model_best.pth')
model.load_state_dict(checkpoint['teacher_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_images)
```

## 🆘 Getting Help

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Read error messages carefully** - they often contain helpful hints
3. **Restart runtime** and try again (Runtime → Restart runtime)
4. **Check GitHub issues** for similar problems
5. **Reduce complexity** - try smaller batch size, fewer epochs

## 🎯 Success Metrics

You know your training is successful when you see:

✅ **GPU detected** and memory allocated  
✅ **Data loaded** with correct number of classes  
✅ **Training progresses** with decreasing loss  
✅ **Validation accuracy** increases over epochs  
✅ **Consistency loss** > 0.001 (for semi-supervised)  
✅ **Model checkpoints** saved successfully  

Target accuracy for fish classification: **75-85%** 

## 🎉 Ready to Start!

Follow the 5 quick start steps above and you'll have your ViT-FishID model training in Google Colab within 10 minutes!

**Happy fish classification! 🐟🚀**
