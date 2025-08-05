# 🚀 Quick Start Guide for Google Colab

This guide helps you get started with ViT-FishID training in Google Colab quickly.

## 📋 Prerequisites Checklist

- [ ] Fish images uploaded to Google Drive
- [ ] Google account (for Colab and Drive access)
- [ ] W&B account (optional, for experiment tracking)

## 🎯 5-Minute Setup

### 1. Organize Your Images in Google Drive

Put your fish images in a folder like this:
```
/MyDrive/Fish_Images/
├── bass_cutout_001.jpg
├── bass_cutout_002.jpg
├── trout_cutout_001.jpg
├── salmon_cutout_001.jpg
└── ...
```

### 2. Open the Colab Notebook

Click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cat-thomson/ViT-FishID/blob/main/Colab_Training.ipynb)

### 3. Enable GPU

- In Colab: Runtime → Change runtime type → Hardware accelerator → GPU → Save

### 4. Update One Line

Find this cell and update the path:
```python
FISH_IMAGES_PATH = f"{DRIVE_ROOT}/Fish_Images"  # ⚠️ UPDATE THIS PATH
```

Change it to match your Google Drive folder:
```python
FISH_IMAGES_PATH = f"{DRIVE_ROOT}/Your_Folder_Name"  # ✅ YOUR PATH HERE
```

### 5. Run All Cells

- Press Ctrl+F9 (or Runtime → Run all)
- The notebook will handle everything automatically!

## 📊 What Happens During Training

1. **Setup** (2-3 minutes)
   - Clones code from GitHub
   - Installs dependencies
   - Mounts Google Drive

2. **Data Organization** (5-10 minutes)
   - Scans your fish images
   - Separates labeled/unlabeled data
   - Creates training structure

3. **Training** (2-3 hours)
   - Trains ViT model with semi-supervised learning
   - Shows progress in real-time
   - Logs metrics to W&B (optional)

4. **Results** (2-3 minutes)
   - Saves trained models
   - Creates summary plots
   - Downloads results to your computer

## 🎛️ Key Configuration Options

You can adjust these in the notebook:

```python
TRAINING_CONFIG = {
    'epochs': 50,        # Increase for better accuracy (50-100)
    'batch_size': 16,    # Decrease if out of memory (8-32)
    'learning_rate': 1e-4,  # Try 5e-5 for more stable training
    # ... other options
}
```

## 🔧 Troubleshooting

### Common Issues & Solutions

**❌ "Path not found" error**
- Check your `FISH_IMAGES_PATH` is correct
- Make sure Google Drive is mounted
- Verify your images are in the specified folder

**❌ "CUDA out of memory" error**
- Reduce `batch_size` to 8 or 12
- Make sure GPU is enabled in runtime settings

**❌ "No images found" error**
- Check image formats (should be .jpg, .jpeg, .png)
- Update the `--labeled_species` list to match your fish types

**❌ W&B login fails**
- This is optional - training will continue without logging
- Get your API key from https://wandb.ai/authorize

### Performance Tips

- **Faster training**: Reduce epochs to 30-40 for quick results
- **Better accuracy**: Increase epochs to 75-100
- **Memory issues**: Reduce batch size to 8-12
- **Speed up data loading**: Organize images in subfolders by species

## 📈 Expected Results

### Training Progress
```
Epoch 1/50: Loss: 3.2 → Acc: 25%
Epoch 10/50: Loss: 2.1 → Acc: 45%
Epoch 25/50: Loss: 1.3 → Acc: 65%
Epoch 50/50: Loss: 0.8 → Acc: 80%
```

### Final Performance
- **Accuracy**: 75-85% (depends on your data quality)
- **Training time**: 2-3 hours on T4 GPU
- **Model size**: ~340MB (ViT-Base)

## 💾 Getting Your Results

The notebook automatically:
1. ✅ Saves best model checkpoint
2. ✅ Creates training plots
3. ✅ Packages everything in a zip file
4. ✅ Downloads to your computer

Your zip file contains:
- `best_model.pth` - Your trained model
- `training_history.json` - Training metrics
- `README.txt` - Usage instructions

## 🚀 Using Your Trained Model

After training, you can use your model like this:

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load your trained model
model = torch.load('best_model.pth', map_location='cpu')

# Preprocess new fish image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('new_fish.jpg')
input_tensor = transform(image).unsqueeze(0)

# Get prediction
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = torch.argmax(outputs, dim=1)
    
print(f"Predicted fish species: {predicted_class}")
```

## 🤝 Need Help?

- 📖 Check the main README.md for detailed documentation
- 🐛 Open an issue on GitHub for bugs
- 💬 Ask questions in the repository discussions

---

🐟 Happy fish classification training! 🚀
