#!/bin/bash

# Example training script for ViT-Base with EMA Teacher-Student framework
# This script demonstrates how to train a Vision Transformer using the EMA framework

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (adjust as needed)
export WANDB_PROJECT="vit-fish-ema"

# Training parameters
DATA_DIR="/path/to/your/fish/dataset"  # UPDATE THIS PATH
MODEL_NAME="vit_base_patch16_224"
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4
EMA_MOMENTUM=0.999
CONSISTENCY_WEIGHT=1.0

# Create output directory
mkdir -p ./outputs/ema_training_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/ema_training_$(date +%Y%m%d_%H%M%S)"

echo "Starting ViT-Base EMA Teacher-Student Training..."
echo "Output directory: $OUTPUT_DIR"

# Run training
python main.py \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --ema_momentum $EMA_MOMENTUM \
    --consistency_weight $CONSISTENCY_WEIGHT \
    --consistency_loss "mse" \
    --temperature 4.0 \
    --image_size 224 \
    --pretrained \
    --warmup_epochs 10 \
    --weight_decay 0.05 \
    --save_dir "$OUTPUT_DIR/checkpoints" \
    --save_frequency 10 \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --seed 42

echo "Training completed! Results saved to: $OUTPUT_DIR"
