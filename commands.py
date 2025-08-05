#!/usr/bin/env python3
"""
🐟 ViT-FishID Command Summary
Quick reference for all available commands in the fish classification pipeline.
"""

import argparse

def print_header():
    print("🐟 ViT-FishID: Semi-Supervised Fish Classification")
    print("=" * 60)
    print()

def print_data_pipeline():
    print("📁 DATA PIPELINE COMMANDS")
    print("-" * 30)
    print()
    
    print("1️⃣ Extract fish cutouts from bounding box annotations:")
    print("   python extract_fish_cutouts.py \\")
    print("       --frames_dir /path/to/frames \\")
    print("       --annotations_dir /path/to/annotations \\")
    print("       --output_dir /path/to/extracted/cutouts \\")
    print("       --buffer_ratio 0.1 \\")
    print("       --min_size 50")
    print()
    
    print("2️⃣ Organize fish images for training:")
    print("   # Interactive mode")
    print("   python organize_fish_data.py \\")
    print("       --input_dir /path/to/fish/cutouts \\")
    print("       --output_dir /path/to/organized/dataset")
    print()
    print("   # Non-interactive mode")
    print("   python organize_fish_data.py \\")
    print("       --input_dir /path/to/fish/cutouts \\")
    print("       --output_dir /path/to/organized/dataset \\")
    print("       --labeled_species bass trout salmon \\")
    print("       --no-interactive")
    print()
    
    print("3️⃣ Complete pipeline (extraction + organization):")
    print("   python fish_pipeline.py \\")
    print("       --frames_dir /path/to/frames \\")
    print("       --annotations_dir /path/to/annotations \\")
    print("       --output_dir /path/to/final/dataset \\")
    print("       --buffer_ratio 0.1 \\")
    print("       --min_cutout_size 50 \\")
    print("       --species_mapping_file species_mapping.json \\")
    print("       --labeled_species bass trout salmon")
    print()

def print_training_commands():
    print("🧠 TRAINING COMMANDS")
    print("-" * 20)
    print()
    
    print("1️⃣ Semi-supervised training (recommended):")
    print("   python main_semi_supervised.py \\")
    print("       --data_dir /path/to/organized/dataset \\")
    print("       --num_epochs 100 \\")
    print("       --batch_size 32 \\")
    print("       --learning_rate 1e-4 \\")
    print("       --use_wandb")
    print()
    
    print("2️⃣ Supervised training (labeled data only):")
    print("   python main.py \\")
    print("       --data_dir /path/to/organized/dataset \\")
    print("       --num_epochs 100 \\")
    print("       --batch_size 32 \\")
    print("       --learning_rate 1e-4")
    print()
    
    print("3️⃣ Resume training from checkpoint:")
    print("   python main_semi_supervised.py \\")
    print("       --data_dir /path/to/dataset \\")
    print("       --resume_from /path/to/checkpoint.pth")
    print()

def print_setup_commands():
    print("⚙️ SETUP COMMANDS")
    print("-" * 17)
    print()
    
    print("1️⃣ Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    
    print("2️⃣ Setup Weights & Biases (optional):")
    print("   wandb login")
    print()
    
    print("3️⃣ Create species mapping file:")
    print("   # Copy and edit example_species_mapping.json")
    print("   cp example_species_mapping.json my_species.json")
    print()

def print_quick_start():
    print("🚀 QUICK START WORKFLOW")
    print("-" * 24)
    print()
    
    print("For users with frame images + bounding box annotations:")
    print()
    print("Step 1: Install dependencies")
    print("   pip install -r requirements.txt")
    print()
    
    print("Step 2: Run complete pipeline")
    print("   python fish_pipeline.py \\")
    print("       --frames_dir /path/to/your/frames \\")
    print("       --annotations_dir /path/to/your/annotations \\")
    print("       --output_dir /path/to/organized/dataset")
    print()
    
    print("Step 3: Train semi-supervised model")
    print("   python main_semi_supervised.py \\")
    print("       --data_dir /path/to/organized/dataset \\")
    print("       --use_wandb")
    print()
    print("That's it! 🎉")
    print()

def print_troubleshooting():
    print("🔧 COMMON ISSUES & SOLUTIONS")
    print("-" * 30)
    print()
    
    print("❌ CUDA out of memory:")
    print("   ✅ Reduce batch size: --batch_size 16")
    print()
    
    print("❌ Poor training performance:")
    print("   ✅ Lower learning rate: --learning_rate 5e-5")
    print("   ✅ Increase consistency weight in config.py")
    print()
    
    print("❌ Data loading errors:")
    print("   ✅ Check file permissions")
    print("   ✅ Verify image formats (JPG, PNG)")
    print("   ✅ Ensure correct directory structure")
    print()
    
    print("❌ Low pseudo-label accuracy:")
    print("   ✅ Increase confidence_threshold in config.py")
    print("   ✅ Train longer on labeled data first")
    print()

def print_file_structure():
    print("📁 EXPECTED DATA STRUCTURE")
    print("-" * 27)
    print()
    
    print("Input (frames + annotations):")
    print("frames/")
    print("├── image001.jpg")
    print("├── image002.jpg")
    print("└── ...")
    print()
    print("annotations/")
    print("├── image001.txt  # YOLO format")
    print("├── image002.txt")
    print("└── ...")
    print()
    
    print("Output (organized dataset):")
    print("organized_dataset/")
    print("├── labeled/")
    print("│   ├── bass/")
    print("│   ├── trout/")
    print("│   └── salmon/")
    print("└── unlabeled/")
    print()

def main():
    parser = argparse.ArgumentParser(description='ViT-FishID Command Reference')
    parser.add_argument('--section', type=str, choices=[
        'all', 'setup', 'data', 'training', 'quickstart', 'troubleshooting', 'structure'
    ], default='all', help='Show specific section')
    
    args = parser.parse_args()
    
    print_header()
    
    if args.section in ['all', 'quickstart']:
        print_quick_start()
        
    if args.section in ['all', 'setup']:
        print_setup_commands()
        
    if args.section in ['all', 'data']:
        print_data_pipeline()
        
    if args.section in ['all', 'training']:
        print_training_commands()
        
    if args.section in ['all', 'structure']:
        print_file_structure()
        
    if args.section in ['all', 'troubleshooting']:
        print_troubleshooting()
    
    if args.section == 'all':
        print("💡 TIP: Use --section to show specific sections only")
        print("   Example: python commands.py --section quickstart")
        print()

if __name__ == '__main__':
    main()
