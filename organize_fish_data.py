#!/usr/bin/env python3
"""
Organize fish cutout images for semi-supervised learning.

This script helps you organize your fish cutout images into the required
directory structure for semi-supervised EMA teacher-student training.

Usage:
    python organize_fish_data.py --input_dir /path/to/fish/cutouts --output_dir /path/to/organized/dataset

Author: GitHub Copilot
Date: 2025
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict
import re


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Organize fish cutout images for semi-supervised learning')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing fish cutout images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for organized dataset')
    parser.add_argument('--labeled_species', type=str, nargs='*', default=None,
                        help='Species names to label (others go to unlabeled). If not provided, interactive selection.')
    parser.add_argument('--copy_files', action='store_true', default=True,
                        help='Copy files instead of moving them (default: True)')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='Use interactive species selection (default: True)')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Disable interactive mode (use with --labeled_species)')
    
    args = parser.parse_args()
    
    # Handle interactive flag
    if args.no_interactive:
        args.interactive = False
    
    return args


def extract_potential_species_names(input_dir: str) -> List[str]:
    """Extract potential species names from filenames."""
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Collect all words from filenames
    all_words = set()
    for image_path in input_path.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            # Extract words from filename
            filename = image_path.stem.lower()
            # Split on common separators and clean
            words = re.split(r'[_\-\s\.]+', filename)
            all_words.update(word for word in words if len(word) > 2)
    
    # Filter out common non-species words
    common_words = {
        'fish', 'image', 'img', 'photo', 'pic', 'picture', 'cutout', 'crop', 'segment',
        'cropped', 'segmented', 'extracted', 'detection', 'result', 'output',
        'final', 'processed', 'clean', 'masked', 'background', 'removed',
        'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp',
        'copy', 'duplicate', 'backup', 'original', 'new', 'old', 'temp',
        'test', 'sample', 'example', 'demo', 'trial'
    }
    
    # Filter out numbers and common words
    potential_species = []
    for word in all_words:
        if (word not in common_words and 
            not word.isdigit() and 
            len(word) > 2 and 
            len(word) < 20):  # Reasonable species name length
            potential_species.append(word)
    
    return sorted(potential_species)


def interactive_species_selection(potential_species: List[str]) -> List[str]:
    """Interactive selection of species to label."""
    print(f"\n🐟 Found {len(potential_species)} potential species names in filenames:")
    print("=" * 60)
    
    # Display in columns
    for i, species in enumerate(potential_species, 1):
        print(f"{i:3d}. {species:<20}", end="")
        if i % 3 == 0:  # 3 columns
            print()
    if len(potential_species) % 3 != 0:
        print()
    
    print("\n" + "=" * 60)
    print("🎯 Select species to LABEL (others will be unlabeled for semi-supervised learning)")
    print("💡 Examples of good species names: salmon, trout, bass, cod, tuna, etc.")
    print("\nOptions:")
    print("  - Enter numbers separated by commas: 1,3,5")
    print("  - Enter ranges with dashes: 1-5,8,10-12")
    print("  - Enter 'none' to manually specify species names")
    print("  - Enter 'all' to label all potential species")
    
    while True:
        try:
            selection = input("\nYour selection: ").strip().lower()
            
            if selection == 'none':
                print("\nEnter species names manually (one per line, empty line to finish):")
                manual_species = []
                while True:
                    species = input("Species name: ").strip()
                    if not species:
                        break
                    manual_species.append(species)
                return manual_species
            
            elif selection == 'all':
                return potential_species
            
            else:
                # Parse selection
                selected_indices = []
                for part in selection.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start-1, end))
                    else:
                        selected_indices.append(int(part)-1)
                
                selected_species = [potential_species[i] for i in selected_indices 
                                  if 0 <= i < len(potential_species)]
                
                if selected_species:
                    print(f"\n✅ Selected species for labeling: {selected_species}")
                    confirm = input("Confirm selection? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return selected_species
                else:
                    print("❌ No valid species selected. Please try again.")
        
        except (ValueError, IndexError) as e:
            print(f"❌ Invalid selection: {e}. Please try again.")


def organize_fish_images(
    input_dir: str,
    output_dir: str,
    labeled_species: List[str],
    copy_files: bool = True
) -> Dict[str, int]:
    """
    Organize fish images into labeled and unlabeled directories.
    
    Returns:
        Dictionary with organization statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    labeled_dir = output_path / "labeled"
    unlabeled_dir = output_path / "unlabeled"
    
    labeled_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    
    # Create species subdirectories
    for species in labeled_species:
        (labeled_dir / species).mkdir(exist_ok=True)
    
    print(f"\n📁 Organizing fish images...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Labeled species: {labeled_species}")
    print(f"  Operation: {'Copy' if copy_files else 'Move'}")
    
    # Statistics
    stats = {
        'total_images': 0,
        'labeled_images': 0,
        'unlabeled_images': 0,
        'species_counts': {species: 0 for species in labeled_species},
        'skipped_files': 0
    }
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Process all images
    for image_path in input_path.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            stats['total_images'] += 1
            
            # Determine species from filename
            filename_lower = image_path.name.lower()
            species_found = None
            
            # Check if filename contains any labeled species name
            for species in labeled_species:
                if species.lower() in filename_lower:
                    species_found = species
                    break
            
            try:
                if species_found:
                    # Move to labeled directory
                    target_path = labeled_dir / species_found / image_path.name
                    stats['labeled_images'] += 1
                    stats['species_counts'][species_found] += 1
                else:
                    # Move to unlabeled directory
                    target_path = unlabeled_dir / image_path.name
                    stats['unlabeled_images'] += 1
                
                # Ensure target directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle file conflicts
                if target_path.exists():
                    counter = 1
                    stem = target_path.stem
                    suffix = target_path.suffix
                    while target_path.exists():
                        target_path = target_path.parent / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                # Copy or move file
                if copy_files:
                    shutil.copy2(image_path, target_path)
                else:
                    shutil.move(str(image_path), str(target_path))
                
                # Progress indicator
                if stats['total_images'] % 100 == 0:
                    print(f"  Processed {stats['total_images']} images...")
                    
            except Exception as e:
                print(f"⚠️ Error processing {image_path}: {e}")
                stats['skipped_files'] += 1
    
    return stats


def save_dataset_info(output_dir: str, labeled_species: List[str], stats: Dict[str, int]):
    """Save dataset information to file."""
    info_path = Path(output_dir) / "dataset_info.txt"
    
    with open(info_path, 'w') as f:
        f.write("🐟 Fish Dataset Organization Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total images processed: {stats['total_images']:,}\n")
        f.write(f"Labeled images: {stats['labeled_images']:,}\n")
        f.write(f"Unlabeled images: {stats['unlabeled_images']:,}\n")
        f.write(f"Skipped files: {stats['skipped_files']:,}\n\n")
        
        f.write("📊 Species Distribution:\n")
        f.write("-" * 25 + "\n")
        for species, count in stats['species_counts'].items():
            f.write(f"{species}: {count:,} images\n")
        
        f.write(f"\n📂 Directory Structure:\n")
        f.write(f"  {output_dir}/\n")
        f.write(f"  ├── labeled/\n")
        for species in labeled_species:
            f.write(f"  │   ├── {species}/\n")
        f.write(f"  └── unlabeled/\n")
        
        f.write(f"\n🚀 Ready for Semi-Supervised Training!\n")
        f.write(f"Run: python main_semi_supervised.py --data_dir {output_dir}\n")
    
    print(f"📄 Dataset info saved to: {info_path}")


def main():
    """Main organization function."""
    args = parse_arguments()
    
    print("🐟 Fish Dataset Organizer for Semi-Supervised Learning")
    print("=" * 60)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return
    
    # Extract potential species names
    print(f"\n🔍 Analyzing filenames in {args.input_dir}...")
    potential_species = extract_potential_species_names(args.input_dir)
    
    if not potential_species:
        print("❌ No potential species names found in filenames.")
        print("💡 Make sure your fish images have species names in their filenames.")
        return
    
    # Select species to label
    if args.labeled_species and not args.interactive:
        labeled_species = args.labeled_species
        print(f"✅ Using provided species list: {labeled_species}")
    else:
        labeled_species = interactive_species_selection(potential_species)
    
    if not labeled_species:
        print("❌ No species selected for labeling. Exiting.")
        return
    
    # Organize images
    stats = organize_fish_images(
        args.input_dir,
        args.output_dir,
        labeled_species,
        args.copy_files
    )
    
    # Print results
    print(f"\n✅ Organization completed!")
    print(f"📊 Results:")
    print(f"  - Total images: {stats['total_images']:,}")
    print(f"  - Labeled images: {stats['labeled_images']:,}")
    print(f"  - Unlabeled images: {stats['unlabeled_images']:,}")
    print(f"  - Skipped files: {stats['skipped_files']:,}")
    
    print(f"\n🏷️ Species distribution:")
    for species, count in stats['species_counts'].items():
        print(f"  - {species}: {count:,} images")
    
    # Save dataset info
    save_dataset_info(args.output_dir, labeled_species, stats)
    
    print(f"\n🚀 Ready for semi-supervised training!")
    print(f"Run: python main_semi_supervised.py --data_dir {args.output_dir}")


if __name__ == '__main__':
    main()
