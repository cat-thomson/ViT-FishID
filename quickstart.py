#!/usr/bin/env python3
"""
Quick Start Script for ViT-FishID.

This script provides an easy way to get started with the project.
It shows example commands and helps validate the installation.

Usage:
    python quickstart.py --help
    python quickstart.py --check-setup
    python quickstart.py --example-commands

Author: GitHub Copilot
Date: 2025
"""

import argparse
import os
import sys
from pathlib import Path


def check_setup():
    """Check if the project is set up correctly."""
    print("🔍 Checking ViT-FishID setup...")
    
    # Check required files
    required_files = [
        'train.py', 'model.py', 'trainer.py', 'data.py', 
        'pipeline.py', 'utils.py', 'requirements.txt', 'species_mapping.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All core files present")
    
    # Check if data directories exist
    data_dirs = ['fish_cutouts', 'Frames']
    for dir_name in data_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Data directory found: {dir_name}")
        else:
            print(f"ℹ️  Data directory not found: {dir_name} (optional)")
    
    # Try importing core modules
    try:
        print("🧪 Testing imports...")
        # We can't actually import without the environment, but we can check syntax
        with open('model.py') as f:
            compile(f.read(), 'model.py', 'exec')
        with open('train.py') as f:
            compile(f.read(), 'train.py', 'exec')
        print("✅ Python files compile successfully")
    except Exception as e:
        print(f"❌ Syntax error in Python files: {e}")
        return False
    
    print("\n🎉 Setup check complete! Project is ready to use.")
    return True


def show_example_commands():
    """Show example commands for common tasks."""
    print("🚀 ViT-FishID Example Commands")
    print("=" * 50)
    
    print("\n📦 1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n⚙️  2. Process Video Data:")
    print("   # Process multiple video directories")
    print("   python pipeline.py \\")
    print("       --multi_video_dir ./Frames \\")
    print("       --output_dir ./data/organized \\")
    print("       --labeled_families Sparidae Serranidae")
    
    print("\n   # Process single directory")
    print("   python pipeline.py \\")
    print("       --frames_dir ./path/to/frames \\")
    print("       --annotations_dir ./path/to/annotations \\")
    print("       --output_dir ./data/organized")
    
    print("\n🚀 3. Train Models:")
    print("   # Semi-supervised training (recommended)")
    print("   python train.py \\")
    print("       --data_dir ./data/organized \\")
    print("       --mode semi_supervised \\")
    print("       --epochs 100 \\")
    print("       --batch_size 32 \\")
    print("       --consistency_weight 2.0 \\")
    print("       --use_wandb")
    
    print("\n   # Supervised training")
    print("   python train.py \\")
    print("       --data_dir ./data/organized \\")
    print("       --mode supervised \\")
    print("       --epochs 100 \\")
    print("       --batch_size 32")
    
    print("\n📊 4. Evaluate Model:")
    print("   python evaluate.py \\")
    print("       --model_path ./checkpoints/model_best.pth \\")
    print("       --data_dir ./data/organized")
    
    print("\n🌐 5. Google Colab:")
    print("   # Open the provided Colab notebook")
    print("   # Upload your data to Google Drive")
    print("   # Run all cells in the notebook")
    
    print("\n💡 6. Common Options:")
    print("   --help                    Show help for any script")
    print("   --preview_only            Preview data processing without saving")
    print("   --verbose                 Show detailed output")
    print("   --use_wandb               Enable Weights & Biases logging")
    
    print("\n📚 7. Documentation:")
    print("   README.md                 Complete project documentation")
    print("   PROJECT_CLEANUP_SUMMARY.md  Project simplification details")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ViT-FishID Quick Start')
    parser.add_argument('--check-setup', action='store_true',
                        help='Check if project is set up correctly')
    parser.add_argument('--example-commands', action='store_true',
                        help='Show example commands')
    
    args = parser.parse_args()
    
    if args.check_setup:
        check_setup()
    elif args.example_commands:
        show_example_commands()
    else:
        print("🐟 ViT-FishID Quick Start")
        print("=" * 30)
        print("\nThis project provides semi-supervised fish classification using Vision Transformers.")
        print("\nQuick options:")
        print("  --check-setup        Verify project setup")
        print("  --example-commands   Show usage examples")
        print("  --help              Show this help message")
        print("\n📚 For complete documentation, see README.md")
        print("🚀 To get started: python quickstart.py --example-commands")


if __name__ == '__main__':
    main()
