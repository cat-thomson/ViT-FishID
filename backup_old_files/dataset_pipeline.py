#!/usr/bin/env python3
"""
Complete pipeline: Extract fish cutouts and organize for semi-supervised learning.

This script combines fish cutout extraction with dataset organization for
seamless preparation of ViT training data.

Usage:
    python fish_pipeline.py --frames_dir /path/to/frames --annotations_dir /path/to/annotations --output_dir /path/to/final/dataset

Author: GitHub Copilot
Date: 2025
"""

import argparse
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Complete fish cutout extraction and organization pipeline')
    
    # Input arguments
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing frame images')
    parser.add_argument('--annotations_dir', type=str, required=True,
                        help='Directory containing YOLO annotation txt files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Final output directory for organized dataset')
    
    # Species mapping arguments
    parser.add_argument('--species_mapping_file', type=str, default=None,
                        help='JSON file with species ID to name mapping')
    parser.add_argument('--full_final_frames_dir', type=str, default=None,
                        help='Directory to scan for species names (auto-create mapping)')
    
    # Extraction arguments
    parser.add_argument('--buffer_ratio', type=float, default=0.15,
                        help='Buffer around bounding box as ratio (default: 0.15)')
    parser.add_argument('--min_size', type=int, default=48,
                        help='Minimum cutout size in pixels (default: 48)')
    parser.add_argument('--max_size', type=int, default=384,
                        help='Maximum cutout size in pixels (default: 384)')
    parser.add_argument('--image_quality', type=int, default=95,
                        help='JPEG quality for saved cutouts (default: 95)')
    
    # Organization arguments
    parser.add_argument('--labeled_species', type=str, nargs='*', default=None,
                        help='Species to label (others go to unlabeled)')
    parser.add_argument('--interactive_selection', action='store_true', default=True,
                        help='Interactively select species to label (default: True)')
    parser.add_argument('--unlabeled_threshold', type=int, default=20,
                        help='Species with fewer images go to unlabeled (default: 20)')
    
    # Pipeline control
    parser.add_argument('--keep_intermediate', action='store_true',
                        help='Keep intermediate cutouts directory')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip extraction if cutouts already exist')
    
    return parser.parse_args()


def run_extraction(args, temp_cutouts_dir: str) -> bool:
    """Run fish cutout extraction."""
    print("🎣 Step 1: Extracting fish cutouts from bounding boxes...")
    
    # Build extraction command
    extraction_cmd = [
        sys.executable, 'extract_fish_cutouts.py',
        '--frames_dir', args.frames_dir,
        '--annotations_dir', args.annotations_dir,
        '--output_dir', temp_cutouts_dir,
        '--buffer_ratio', str(args.buffer_ratio),
        '--min_size', str(args.min_size),
        '--max_size', str(args.max_size),
        '--image_quality', str(args.image_quality)
    ]
    
    # Add species mapping arguments
    if args.species_mapping_file:
        extraction_cmd.extend(['--species_mapping_file', args.species_mapping_file])
    elif args.full_final_frames_dir:
        extraction_cmd.extend([
            '--create_species_mapping',
            '--full_final_frames_dir', args.full_final_frames_dir
        ])
    
    try:
        result = subprocess.run(extraction_cmd, check=True, capture_output=True, text=True)
        print("✅ Fish cutout extraction completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Extraction failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def run_organization(args, cutouts_dir: str) -> bool:
    """Run dataset organization for semi-supervised learning."""
    print("\n📁 Step 2: Organizing cutouts for semi-supervised learning...")
    
    # Build organization command
    organization_cmd = [
        sys.executable, 'organize_fish_data.py',
        '--input_dir', cutouts_dir,
        '--output_dir', args.output_dir,
        '--copy_files'  # Copy instead of move to preserve originals
    ]
    
    # Add species selection arguments
    if args.labeled_species and not args.interactive_selection:
        organization_cmd.extend(['--labeled_species'] + args.labeled_species)
        organization_cmd.append('--no-interactive')
    elif args.interactive_selection:
        organization_cmd.append('--interactive')
    
    try:
        result = subprocess.run(organization_cmd, check=True, capture_output=True, text=True)
        print("✅ Dataset organization completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Organization failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def analyze_cutouts_directory(cutouts_dir: str) -> dict:
    """Analyze extracted cutouts and provide statistics."""
    stats = {'total_cutouts': 0, 'species_counts': {}, 'recommended_labeled': []}
    
    cutouts_path = Path(cutouts_dir)
    
    for species_dir in cutouts_path.iterdir():
        if species_dir.is_dir():
            count = len(list(species_dir.glob('*.jpg')))
            stats['species_counts'][species_dir.name] = count
            stats['total_cutouts'] += count
            
            # Recommend species with sufficient samples for labeling
            if count >= 50:  # Good for labeling
                stats['recommended_labeled'].append(species_dir.name)
    
    return stats


def smart_species_selection(cutouts_dir: str, min_labeled_samples: int = 30) -> list:
    """Automatically select species suitable for labeling."""
    stats = analyze_cutouts_directory(cutouts_dir)
    
    # Select species with enough samples and clear names
    selected_species = []
    
    for species, count in stats['species_counts'].items():
        if (count >= min_labeled_samples and 
            len(species) > 3 and 
            not species.startswith('unknown_') and
            species.replace('_', '').isalpha()):
            selected_species.append(species)
    
    return sorted(selected_species)


def main():
    """Main pipeline function."""
    args = parse_arguments()
    
    print("🐟 Fish Cutout Extraction and Organization Pipeline")
    print("=" * 60)
    print(f"📂 Frames directory: {args.frames_dir}")
    print(f"📝 Annotations directory: {args.annotations_dir}")
    print(f"🎯 Final output directory: {args.output_dir}")
    
    # Validate input directories
    if not os.path.exists(args.frames_dir):
        print(f"❌ Frames directory does not exist: {args.frames_dir}")
        return
    
    if not os.path.exists(args.annotations_dir):
        print(f"❌ Annotations directory does not exist: {args.annotations_dir}")
        return
    
    # Create temporary directory for cutouts
    temp_cutouts_dir = None
    
    try:
        if args.keep_intermediate:
            temp_cutouts_dir = os.path.join(os.path.dirname(args.output_dir), "fish_cutouts_temp")
            Path(temp_cutouts_dir).mkdir(parents=True, exist_ok=True)
        else:
            temp_cutouts_dir = tempfile.mkdtemp(prefix="fish_cutouts_")
        
        print(f"🔧 Using temporary cutouts directory: {temp_cutouts_dir}")
        
        # Step 1: Extract fish cutouts
        if not args.skip_extraction or not os.path.exists(temp_cutouts_dir):
            success = run_extraction(args, temp_cutouts_dir)
            if not success:
                print("❌ Pipeline failed at extraction step")
                return
        else:
            print("⏭️ Skipping extraction (using existing cutouts)")
        
        # Analyze extracted cutouts
        print("\n📊 Analyzing extracted cutouts...")
        stats = analyze_cutouts_directory(temp_cutouts_dir)
        
        print(f"📈 Extraction Statistics:")
        print(f"  - Total cutouts: {stats['total_cutouts']:,}")
        print(f"  - Species found: {len(stats['species_counts'])}")
        print(f"  - Recommended for labeling: {len(stats['recommended_labeled'])}")
        
        print(f"\n🐟 Species breakdown:")
        for species, count in sorted(stats['species_counts'].items(), key=lambda x: x[1], reverse=True):
            status = "📌 Good for labeling" if count >= 50 else "📍 Consider for unlabeled"
            print(f"  - {species}: {count:,} cutouts {status}")
        
        # Smart species selection for semi-supervised learning
        if not args.labeled_species and not args.interactive_selection:
            print(f"\n🧠 Smart species selection for semi-supervised learning...")
            recommended_species = smart_species_selection(temp_cutouts_dir, args.unlabeled_threshold)
            
            if recommended_species:
                print(f"✅ Recommended species for labeling: {recommended_species}")
                args.labeled_species = recommended_species
                args.interactive_selection = False
            else:
                print("⚠️ No species meet criteria for automatic selection. Using interactive mode.")
                args.interactive_selection = True
        
        # Step 2: Organize for semi-supervised learning
        success = run_organization(args, temp_cutouts_dir)
        if not success:
            print("❌ Pipeline failed at organization step")
            return
        
        # Final summary
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Final dataset location: {args.output_dir}")
        
        # Check final organization
        labeled_dir = Path(args.output_dir) / "labeled"
        unlabeled_dir = Path(args.output_dir) / "unlabeled"
        
        if labeled_dir.exists():
            labeled_species = [d.name for d in labeled_dir.iterdir() if d.is_dir()]
            labeled_count = sum(len(list((labeled_dir / species).glob('*.jpg'))) for species in labeled_species)
            print(f"🏷️ Labeled data: {labeled_count:,} images across {len(labeled_species)} species")
            print(f"   Species: {', '.join(labeled_species)}")
        
        if unlabeled_dir.exists():
            unlabeled_count = len(list(unlabeled_dir.glob('*.jpg')))
            print(f"🔍 Unlabeled data: {unlabeled_count:,} images")
        
        # Next steps
        print(f"\n🚀 Ready for ViT training!")
        print(f"Run: python main_semi_supervised.py --data_dir {args.output_dir}")
        
        # Generate training command suggestion
        suggested_cmd = f"""
# Suggested training command:
python main_semi_supervised.py \\
    --data_dir {args.output_dir} \\
    --model_name vit_base_patch16_224 \\
    --batch_size 32 \\
    --epochs 150 \\
    --unlabeled_ratio 3.0 \\
    --consistency_weight 2.0 \\
    --ema_momentum 0.999 \\
    --use_wandb
"""
        print(suggested_cmd)
        
        # Save command to file
        cmd_file = Path(args.output_dir) / "suggested_training_command.sh"
        with open(cmd_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated training command for your fish dataset\n\n")
            f.write(suggested_cmd.strip())
        
        print(f"💾 Training command saved to: {cmd_file}")
    
    finally:
        # Clean up temporary directory if not keeping
        if temp_cutouts_dir and not args.keep_intermediate:
            try:
                shutil.rmtree(temp_cutouts_dir)
                print(f"🗑️ Cleaned up temporary directory")
            except:
                print(f"⚠️ Could not clean up temporary directory: {temp_cutouts_dir}")


if __name__ == '__main__':
    main()
