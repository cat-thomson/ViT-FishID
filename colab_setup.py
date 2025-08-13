#!/usr/bin/env python3
"""
Google Colab Setup Helper for ViT-FishID
This script provides utilities to help set up training in Google Colab.
"""

import os
import json
import subprocess
import zipfile
import shutil
import time
from typing import Dict, List, Optional

def check_colab_environment() -> Dict[str, str]:
    """Check if running in Google Colab and return environment info."""
    env_info = {
        'is_colab': False,
        'python_version': os.sys.version,
        'gpu_available': False,
        'gpu_name': 'None',
        'gpu_memory': '0 GB'
    }
    
    # Check if running in Colab
    try:
        import sys
        env_info['is_colab'] = 'google.colab' in sys.modules
    except:
        pass
    
    try:
        import torch
        env_info['pytorch_version'] = torch.__version__
        env_info['gpu_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            env_info['gpu_name'] = torch.cuda.get_device_name(0)
            env_info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    except ImportError:
        env_info['pytorch_version'] = 'Not installed'
    
    return env_info

def install_dependencies() -> bool:
    """Install all required dependencies for ViT-FishID."""
    packages = [
        'torch', 'torchvision', 'torchaudio',
        'timm', 'transformers',
        'albumentations',
        'wandb',
        'opencv-python-headless',
        'scikit-learn',
        'matplotlib', 'seaborn',
        'tqdm'
    ]
    
    print("📦 Installing ViT-FishID dependencies...")
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call(['pip', 'install', '-q', package])
        
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def validate_data_structure(data_dir: str) -> Dict[str, any]:
    """Validate the data directory structure."""
    validation_result = {
        'valid': False,
        'structure_type': 'unknown',
        'num_classes': 0,
        'labeled_samples': 0,
        'unlabeled_samples': 0,
        'classes': [],
        'errors': []
    }
    
    if not os.path.exists(data_dir):
        validation_result['errors'].append(f"Data directory does not exist: {data_dir}")
        return validation_result
    
    # Check for semi-supervised structure
    labeled_dir = os.path.join(data_dir, 'labeled')
    unlabeled_dir = os.path.join(data_dir, 'unlabeled')
    
    if os.path.exists(labeled_dir) and os.path.exists(unlabeled_dir):
        validation_result['structure_type'] = 'semi_supervised'
        validation_result['valid'] = True
        
        # Count classes and samples in labeled directory
        classes = [d for d in os.listdir(labeled_dir) 
                  if os.path.isdir(os.path.join(labeled_dir, d))]
        validation_result['classes'] = classes
        validation_result['num_classes'] = len(classes)
        
        # Count labeled samples
        labeled_count = 0
        for class_dir in classes:
            class_path = os.path.join(labeled_dir, class_dir)
            if os.path.isdir(class_path):
                class_samples = len([f for f in os.listdir(class_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                labeled_count += class_samples
        
        validation_result['labeled_samples'] = labeled_count
        
        # Count unlabeled samples
        unlabeled_files = [f for f in os.listdir(unlabeled_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        validation_result['unlabeled_samples'] = len(unlabeled_files)
        
    elif any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir)):
        validation_result['structure_type'] = 'supervised'
        validation_result['valid'] = True
        
        # Count classes for supervised structure
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        validation_result['classes'] = classes
        validation_result['num_classes'] = len(classes)
        
        # Count samples
        labeled_count = 0
        for class_dir in classes:
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                class_samples = len([f for f in os.listdir(class_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                labeled_count += class_samples
        
        validation_result['labeled_samples'] = labeled_count
        
    else:
        validation_result['errors'].append("No valid data structure found")
        validation_result['errors'].append("Expected: labeled/ and unlabeled/ subdirectories")
    
    return validation_result

def generate_training_command(config: Dict[str, any]) -> str:
    """Generate the training command with given configuration."""
    cmd_parts = [
        "python train.py",
        f"--data_dir \"{config['data_dir']}\"",
        f"--mode {config['mode']}",
        f"--epochs {config['epochs']}",
        f"--batch_size {config['batch_size']}",
        f"--image_size {config['image_size']}",
        f"--num_workers {config['num_workers']}",
        f"--val_split {config['val_split']}",
        f"--test_split {config['test_split']}",
        f"--model_name {config['model_name']}",
        f"--learning_rate {config['learning_rate']}",
        f"--weight_decay {config['weight_decay']}",
        f"--warmup_epochs {config['warmup_epochs']}",
        f"--consistency_weight {config['consistency_weight']}",
        f"--pseudo_label_threshold {config['pseudo_label_threshold']}",
        f"--temperature {config['temperature']}",
        f"--unlabeled_ratio {config['unlabeled_ratio']}",
        f"--ramp_up_epochs {config['ramp_up_epochs']}",
        f"--ema_momentum {config['ema_momentum']}",
        f"--save_frequency {config['save_frequency']}",
        f"--seed {config['seed']}"
    ]
    
    if config.get('pretrained', True):
        cmd_parts.append("--pretrained")
    
    if config.get('use_wandb', False):
        cmd_parts.append("--use_wandb")
        cmd_parts.append(f"--wandb_project {config['wandb_project']}")
    
    return " \\\n    ".join(cmd_parts)

def get_colab_optimal_config(gpu_memory_gb: float) -> Dict[str, any]:
    """Get optimal training configuration for Colab based on GPU memory."""
    if gpu_memory_gb >= 14:  # T4 or better
        batch_size = 16
        num_workers = 2
    elif gpu_memory_gb >= 10:
        batch_size = 12
        num_workers = 2
    else:
        batch_size = 8
        num_workers = 1
    
    return {
        'mode': 'semi_supervised',
        'epochs': 50,  # Reduced for Colab
        'batch_size': batch_size,
        'image_size': 224,
        'num_workers': num_workers,
        'val_split': 0.2,
        'test_split': 0.2,
        'model_name': 'vit_base_patch16_224',
        'pretrained': True,
        'dropout_rate': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'warmup_epochs': 5,
        'consistency_weight': 2.0,
        'pseudo_label_threshold': 0.7,
        'temperature': 4.0,
        'unlabeled_ratio': 2.0,
        'ramp_up_epochs': 10,
        'ema_momentum': 0.999,
        'save_frequency': 10,
        'seed': 42,
        'use_wandb': False,  # User can enable if desired
        'wandb_project': 'vit-fish-colab'
    }

def extract_dataset_zip(zip_path: str, extract_to: str = '/content/fish_cutouts') -> Dict[str, any]:
    """
    Extract fish dataset ZIP file and validate structure.
    
    Args:
        zip_path: Path to the ZIP file in Google Drive
        extract_to: Local path to extract the dataset
        
    Returns:
        Dictionary with extraction results and validation info
    """
    result = {
        'success': False,
        'extract_path': extract_to,
        'zip_size_mb': 0,
        'extraction_time': 0,
        'num_files': 0,
        'data_structure': {},
        'errors': []
    }
    
    # Check if ZIP file exists
    if not os.path.exists(zip_path):
        result['errors'].append(f"ZIP file not found: {zip_path}")
        return result
    
    try:
        # Get ZIP file info
        result['zip_size_mb'] = os.path.getsize(zip_path) / (1024 * 1024)
        
        # Remove existing extracted data if present
        if os.path.exists(extract_to):
            shutil.rmtree(extract_to)
        
        # Extract ZIP file
        start_time = time.time()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temporary location first
            temp_extract_dir = '/content/temp_extract'
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
            
            zip_ref.extractall(temp_extract_dir)
            
            # Find the actual dataset directory inside the extracted content
            extracted_items = os.listdir(temp_extract_dir)
            
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract_dir, extracted_items[0])):
                # ZIP contains a single directory - move its contents
                inner_dir = os.path.join(temp_extract_dir, extracted_items[0])
                shutil.move(inner_dir, extract_to)
            else:
                # ZIP contains multiple items at root level - move temp dir
                shutil.move(temp_extract_dir, extract_to)
            
            # Clean up temp directory if it still exists
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
        
        result['extraction_time'] = time.time() - start_time
        result['success'] = True
        
        # Validate extracted data
        validation = validate_data_structure(extract_to)
        result['data_structure'] = validation
        
        # Count total files
        total_files = 0
        for root, dirs, files in os.walk(extract_to):
            total_files += len(files)
        result['num_files'] = total_files
        
    except zipfile.BadZipFile:
        result['errors'].append("Invalid ZIP file format")
    except Exception as e:
        result['errors'].append(f"Extraction error: {str(e)}")
    
    return result

def print_setup_status():
    """Print comprehensive setup status for Colab."""
    print("🐟 ViT-FishID Google Colab Setup Helper")
    print("=" * 50)
    
    # Check environment
    env_info = check_colab_environment()
    
    print("\n🔍 Environment Check:")
    print(f"  - Running in Colab: {'✅ Yes' if env_info['is_colab'] else '❌ No'}")
    print(f"  - Python: {env_info['python_version'].split()[0]}")
    print(f"  - PyTorch: {env_info['pytorch_version']}")
    print(f"  - GPU Available: {'✅ Yes' if env_info['gpu_available'] else '❌ No'}")
    
    if env_info['gpu_available']:
        print(f"  - GPU: {env_info['gpu_name']}")
        print(f"  - GPU Memory: {env_info['gpu_memory']}")
        
        # Get optimal config
        gpu_memory = float(env_info['gpu_memory'].split()[0])
        optimal_config = get_colab_optimal_config(gpu_memory)
        
        print(f"\n💡 Recommended Settings for your GPU:")
        print(f"  - Batch Size: {optimal_config['batch_size']}")
        print(f"  - Epochs: {optimal_config['epochs']} (2-3 hours)")
        print(f"  - Workers: {optimal_config['num_workers']}")
    else:
        print("\n⚠️  No GPU detected! Enable GPU runtime:")
        print("   Runtime → Change runtime type → Hardware accelerator → GPU")

if __name__ == "__main__":
    print_setup_status()
