# 🐟 Google Colab Setup Guide - Semi-Supervised ViT Training

## 📋 Prerequisites

### 1. **Upload Fish Dataset to Google Drive**
```
📁 Google Drive/
├── fish_dataset.zip (recommended)
└── OR fish_images/
    ├── species1/
    ├── species2/
    └── ...
```

### 2. **Get GPU Runtime**
- In Colab: **Runtime → Change runtime type → GPU → T4/V100**

## 🚀 Quick Start (5 Steps)

### **Step 1: Open Colab Notebook**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `Colab_Semi_Supervised_Training.ipynb`
3. Or connect to GitHub and import from `cat-thomson/ViT-FishID`

### **Step 2: Update Data Path**
In **Step 3** of the notebook, update this line:
```python
DRIVE_DATA_PATH = "/content/drive/MyDrive/YOUR_FISH_DATASET.zip"  # 👈 UPDATE THIS
```

### **Step 3: Run All Cells**
- **Runtime → Run all** (or Ctrl+F9)
- **Or** run each cell individually with Shift+Enter

### **Step 4: Monitor Training**
Watch for these **success indicators**:
```
✅ Cons Loss: 0.028 (NOT 0.0000)
✅ High-conf Pseudo: 847/4000 (21.2%)
✅ Validation Accuracy: 67.3% (up from 60%)
```

### **Step 5: Download Results**
The notebook will automatically:
- Package your trained model
- Download to your computer
- Backup to Google Drive

## 📊 Expected Timeline

| Phase | Duration | What Happens |
|-------|----------|--------------|
| **Setup** | 5-10 min | Install packages, mount Drive, clone repo |
| **Data Prep** | 5-15 min | Extract/organize 17K fish images |
| **Training** | 2-4 hours | 50 epochs with semi-supervised learning |
| **Results** | 5 min | Package and download trained model |

## 🎯 What to Expect

### **Before (Your Current Results)**
```
Consistency Loss: 0.0000 ❌
High-conf Pseudo: 0.0% ❌  
Validation Accuracy: ~60%
Using: 3,273 labeled images only
```

### **After (Fixed Semi-Supervised)**
```
Consistency Loss: 0.001-0.1 ✅
High-conf Pseudo: 10-30% ✅
Validation Accuracy: 65-75% 📈
Using: 3,273 labeled + 13,908 unlabeled images
```

## 🔧 Troubleshooting

### **Problem: "No GPU available"**
**Solution:** Runtime → Change runtime type → Hardware accelerator → GPU

### **Problem: "Dataset not found"**
**Solution:** Update `DRIVE_DATA_PATH` in Step 3 to point to your fish images

### **Problem: "Consistency Loss = 0.0000"**
**Solution:** Already fixed! The new notebook uses optimized parameters:
- Pseudo-label threshold: 0.7 (was 0.95)
- Temperature: 4.0
- MSE consistency loss

### **Problem: "Out of memory"**
**Solution:** In the training cell, reduce batch size:
```python
--batch_size 16  # instead of 32
```

### **Problem: "Training too slow"**
**Solution:** Reduce epochs for testing:
```python
--epochs 20  # instead of 50
```

## 📱 Monitoring Tips

### **W&B Dashboard (Optional)**
If you set up Weights & Biases:
1. Go to [wandb.ai](https://wandb.ai)
2. Look for project: `vit-fish-semi-supervised-colab`
3. Monitor real-time training progress

### **Manual Monitoring**
Watch the training logs for:
```
Epoch 10/50 Train - Total Loss: 1.45, Sup Loss: 1.42, Cons Loss: 0.028
Train - Labeled Acc: 68.4%, Pseudo Acc: 72.1%
Train - High-conf Pseudo: 21.2%
Student Val - Acc: 67.3%
Teacher Val - Acc: 68.1%
```

## 🎉 Success Criteria

✅ **Consistency Loss > 0.001** (semi-supervised active)  
✅ **High-conf Pseudo > 10%** (unlabeled data used)  
✅ **Validation Accuracy > 65%** (improvement over 60%)  
✅ **Steady training progress** (loss decreasing)  

## 📞 Need Help?

### **Quick Fixes**
```python
# If consistency loss still 0.0000:
--pseudo_label_threshold 0.5

# If no pseudo-labels generated:
--temperature 5.0

# If validation accuracy stuck:
--unlabeled_ratio 3.0
```

### **Alternative: Shorter Test Run**
For a quick test (30 minutes):
```python
--epochs 10 --batch_size 16 --save_frequency 5
```

## 🏆 Final Output

You'll get a ZIP file containing:
- 🏆 **Best trained model** (`model_best.pth`)
- 📊 **Training checkpoints** 
- 📝 **Source code**
- 📓 **This notebook**
- ☁️ **Google Drive backup**

**Ready to leverage 13,908 unlabeled fish images for better classification! 🐟**
