#!/usr/bin/env python3
"""
Google Colab Setup Script for ViT-FishID

This script helps configure the ViT-FishID project for running in Google Colab.
It handles path configuration, data organization, and training setup.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def setup_colab_config(drive_images_path, output_path=None):
    """
    Create a configuration file for Google Colab training.
    
    Args:
        drive_images_path (str): Path to fish images in Google Drive
        output_path (str): Path where training outputs will be saved
    """
    
    if output_path is None:
        output_path = f"{os.path.dirname(drive_images_path)}/Fish_Training_Output"
    
    config = {
        "paths": {
            "drive_root": "/content/drive/MyDrive",
            "fish_images": drive_images_path,
            "organized_dataset": f"{output_path}/organized_fish_dataset",
            "checkpoints": f"{output_path}/checkpoints",
            "results": f"{output_path}/results"
        },
        "training": {
            "batch_size": 16,  # Reduced for Colab
            "epochs": 50,      # Reduced for faster training
            "learning_rate": 1e-4,
            "num_workers": 2,  # Reduced for Colab
            "warmup_epochs": 5,
            "ramp_up_epochs": 10,
            "save_frequency": 5
        },
        "model": {
            "name": "vit_base_patch16_224",
            "pretrained": True,
            "dropout_rate": 0.1
        },
        "semi_supervised": {
            "ema_momentum": 0.999,
            "consistency_loss": "mse",
            "consistency_weight": 1.0,
            "pseudo_label_threshold": 0.95,
            "temperature": 4.0,
            "unlabeled_ratio": 2.0
        },
        "wandb": {
            "project": "vit-fish-colab",
            "entity": None  # Will be filled during login
        }
    }
    
    # Save config file
    config_path = "colab_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Colab configuration saved to: {config_path}")
    return config_path

def check_colab_environment():
    """Check if running in Google Colab and verify setup."""
    
    try:
        import google.colab
        print("✅ Running in Google Colab")
        
        # Check if drive is mounted
        if os.path.exists('/content/drive'):
            print("✅ Google Drive appears to be mounted")
        else:
            print("⚠️ Google Drive not mounted. Run: from google.colab import drive; drive.mount('/content/drive')")
        
        # Check GPU
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ No GPU detected. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU")
            
        return True
        
    except ImportError:
        print("❌ Not running in Google Colab")
        return False

def install_colab_dependencies():
    """Install dependencies for Google Colab."""
    
    print("📦 Installing dependencies for Google Colab...")
    
    # Install PyTorch for CUDA
    os.system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Install other dependencies
    dependencies = [
        "timm",
        "transformers", 
        "wandb",
        "pillow",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm"
    ]
    
    for dep in dependencies:
        os.system(f"pip install -q {dep}")
    
    print("✅ Dependencies installed")

def create_colab_training_script(config_path):
    """Create a simplified training script for Colab."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Simplified training script for Google Colab
Generated automatically by setup_colab.py
"""

import os
import json
import torch
import wandb
from datetime import datetime

# Load configuration
with open("{config_path}", "r") as f:
    config = json.load(f)

def run_training():
    """Run semi-supervised training with Colab configuration."""
    
    print("🚀 Starting ViT-FishID training in Google Colab...")
    
    # Build training command
    paths = config["paths"]
    training = config["training"]
    model = config["model"]
    ssl = config["semi_supervised"]
    
    cmd_args = [
        f'--data_dir "{{paths["organized_dataset"]}}"',
        f'--batch_size {{training["batch_size"]}}',
        f'--epochs {{training["epochs"]}}',
        f'--learning_rate {{training["learning_rate"]}}',
        f'--num_workers {{training["num_workers"]}}',
        f'--warmup_epochs {{training["warmup_epochs"]}}',
        f'--ramp_up_epochs {{training["ramp_up_epochs"]}}',
        f'--save_frequency {{training["save_frequency"]}}',
        f'--model_name {{model["name"]}}',
        f'--dropout_rate {{model["dropout_rate"]}}',
        f'--ema_momentum {{ssl["ema_momentum"]}}',
        f'--consistency_loss {{ssl["consistency_loss"]}}',
        f'--consistency_weight {{ssl["consistency_weight"]}}',
        f'--pseudo_label_threshold {{ssl["pseudo_label_threshold"]}}',
        f'--temperature {{ssl["temperature"]}}',
        f'--unlabeled_ratio {{ssl["unlabeled_ratio"]}}',
        f'--save_dir "{{paths["checkpoints"]}}"',
        f'--device cuda',
        f'--use_wandb',
        f'--wandb_project {{config["wandb"]["project"]}}',
        f'--wandb_run_name colab-{{datetime.now().strftime("%Y%m%d-%H%M%S")}}'
    ]
    
    cmd = f"python main_semi_supervised.py {{' '.join(cmd_args)}}"
    
    print("Executing:", cmd)
    os.system(cmd)

if __name__ == "__main__":
    run_training()
'''
    
    script_path = "run_colab_training.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Colab training script created: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Setup ViT-FishID for Google Colab")
    parser.add_argument("--drive_images_path", type=str, required=True,
                        help="Path to fish images in Google Drive (e.g., /content/drive/MyDrive/Fish_Images)")
    parser.add_argument("--output_path", type=str, 
                        help="Path for training outputs (default: same directory as images)")
    parser.add_argument("--check_env", action="store_true",
                        help="Check if environment is properly set up for Colab")
    parser.add_argument("--install_deps", action="store_true",
                        help="Install dependencies for Colab")
    
    args = parser.parse_args()
    
    if args.check_env:
        check_colab_environment()
    
    if args.install_deps:
        install_colab_dependencies()
    
    if args.drive_images_path:
        config_path = setup_colab_config(args.drive_images_path, args.output_path)
        create_colab_training_script(config_path)
        
        print("\n🎉 Google Colab setup complete!")
        print("\nNext steps:")
        print("1. Upload this repository to GitHub (excluding images)")
        print("2. Upload your fish images to Google Drive")
        print("3. Open the Colab_Training.ipynb notebook in Google Colab")
        print("4. Update the FISH_IMAGES_PATH in the notebook")
        print("5. Run all cells to start training")

if __name__ == "__main__":
    main()
