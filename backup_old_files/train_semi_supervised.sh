#!/bin/bash

# Semi-supervised training script for ViT-Base with EMA Teacher-Student framework
# This script demonstrates how to train with both labeled and unlabeled fish images

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (adjust as needed)
export WANDB_PROJECT="vit-fish-semi-supervised"

# Data paths - UPDATE THESE PATHS
FISH_CUTOUTS_DIR="/path/to/your/fish/cutouts"  # Your original fish cutouts
ORGANIZED_DATA_DIR="/path/to/organized/dataset"  # Where organized data will be stored

# Training parameters
MODEL_NAME="vit_base_patch16_224"
BATCH_SIZE=32
EPOCHS=150
LEARNING_RATE=1e-4
EMA_MOMENTUM=0.999
CONSISTENCY_WEIGHT=2.0
UNLABELED_RATIO=3.0
PSEUDO_LABEL_THRESHOLD=0.95

# Create output directory
mkdir -p ./outputs/semi_supervised_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/semi_supervised_$(date +%Y%m%d_%H%M%S)"

echo "🐟 Starting Semi-Supervised ViT-FishID Training..."
echo "Output directory: $OUTPUT_DIR"

# Step 1: Organize fish cutouts (if not already done)
if [ ! -d "$ORGANIZED_DATA_DIR" ]; then
    echo "📁 Organizing fish cutouts..."
    python organize_fish_data.py \
        --input_dir "$FISH_CUTOUTS_DIR" \
        --output_dir "$ORGANIZED_DATA_DIR" \
        --interactive
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to organize fish data. Please check the input directory."
        exit 1
    fi
else
    echo "✅ Using existing organized dataset: $ORGANIZED_DATA_DIR"
fi

# Step 2: Train with semi-supervised learning
echo "🚀 Starting semi-supervised training..."
python main_semi_supervised.py \
    --data_dir "$ORGANIZED_DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --ema_momentum $EMA_MOMENTUM \
    --consistency_weight $CONSISTENCY_WEIGHT \
    --unlabeled_ratio $UNLABELED_RATIO \
    --pseudo_label_threshold $PSEUDO_LABEL_THRESHOLD \
    --consistency_loss "mse" \
    --temperature 4.0 \
    --image_size 224 \
    --pretrained \
    --warmup_epochs 10 \
    --ramp_up_epochs 25 \
    --weight_decay 0.05 \
    --save_dir "$OUTPUT_DIR/checkpoints" \
    --save_frequency 10 \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --seed 42

echo "✅ Semi-supervised training completed! Results saved to: $OUTPUT_DIR"

# Step 3: Evaluate the trained model (optional)
if [ -f "$OUTPUT_DIR/checkpoints/model_best.pth" ]; then
    echo "📊 Evaluating trained model..."
    python evaluate.py \
        --checkpoint_path "$OUTPUT_DIR/checkpoints/model_best.pth" \
        --data_dir "$ORGANIZED_DATA_DIR" \
        --output_dir "$OUTPUT_DIR/evaluation"
fi

echo "🎉 All done! Check the results in $OUTPUT_DIR"
