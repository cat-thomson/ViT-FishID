# ViT-FishID Project Consolidation Plan

## Current State Analysis
- **40+ Python files** with overlapping functionality
- **Multiple trainers**: `ema_trainer.py`, `semi_supervised_trainer.py`
- **Multiple data loaders**: `data_loader.py`, `semi_supervised_data.py`
- **Multiple main scripts**: `main.py`, `main_semi_supervised.py`
- **Multiple pipeline scripts**: `fish_pipeline.py`, `multi_video_pipeline.py`, `dataset_pipeline.py`, `updated_fish_pipeline.py`
- **Scattered documentation**: Multiple README and guide files

## Proposed Simplified Structure

### Core Files (8 essential files):
```
ViT-FishID/
├── README.md                     # Complete updated documentation
├── requirements.txt              # Dependencies
├── train.py                      # Main training script (replaces main.py + main_semi_supervised.py)
├── model.py                      # ViT model + EMA teacher (replaces vit_model.py + ema_teacher.py)
├── trainer.py                    # Unified trainer (replaces ema_trainer.py + semi_supervised_trainer.py)
├── data.py                       # Unified data loading (replaces data_loader.py + semi_supervised_data.py)
├── pipeline.py                   # Data processing pipeline (replaces all pipeline files)
├── utils.py                      # Utilities (keep existing)
```

### Optional Files:
```
├── config.py                     # Configuration (if needed)
├── evaluate.py                   # Model evaluation (keep if used)
├── Colab_Training.ipynb          # Google Colab notebook
├── species_mapping.txt           # Species mapping file
```

### Folders:
```
├── data/                         # Data files
├── checkpoints/                  # Model checkpoints
├── outputs/                      # Training outputs
```

## Files to Remove:
- `ema_trainer.py` → Merge into `trainer.py`
- `semi_supervised_trainer.py` → Merge into `trainer.py`
- `main.py` → Merge into `train.py`
- `main_semi_supervised.py` → Merge into `train.py`
- `data_loader.py` → Merge into `data.py`
- `semi_supervised_data.py` → Merge into `data.py`
- `vit_model.py` → Merge into `model.py`
- `ema_teacher.py` → Merge into `model.py`
- All redundant pipeline files → Keep only `pipeline.py`
- Multiple documentation files → Keep only updated `README.md`
- Test/demo files that aren't essential
- Shell scripts (can be recreated from README examples)

## Implementation Plan:
1. Create unified `model.py` with ViT + EMA
2. Create unified `trainer.py` with both supervised and semi-supervised training
3. Create unified `data.py` with all data loading functionality
4. Create unified `train.py` with both training modes
5. Create unified `pipeline.py` for data processing
6. Update `README.md` with complete documentation
7. Remove redundant files
8. Test the simplified structure

## Benefits:
- **Reduced complexity**: 8 core files instead of 40+
- **Single entry point**: `train.py` for all training
- **Cleaner structure**: Clear separation of concerns
- **Easier maintenance**: Less duplication
- **Better documentation**: Consolidated in README
