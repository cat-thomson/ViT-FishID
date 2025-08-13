#!/usr/bin/env python3
"""
Unified Data Pipeline for ViT-FishID.

This script handles:
- Extracting fish cutouts from bounding box annotations (YOLO format)
- Organizing cutouts into labeled/unlabeled structure for training
- Supporting multiple video directories and species mapping

Usage:
    # Extract and organize fish cutouts
    python pipeline.py --frames_dir /path/to/frames --annotations_dir /path/to/annotations --output_dir /path/to/dataset
    
    # Process multiple video directories
    python pipeline.py --multi_video_dir /path/to/video/directories --output_dir /path/to/dataset

Author: GitHub Copilot
Date: 2025
"""

import os
import argparse
import json
import re
import cv2
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ViT-FishID Data Pipeline')
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--frames_dir', type=str,
                       help='Directory containing frame images')
    group.add_argument('--multi_video_dir', type=str,
                       help='Directory containing multiple video subdirectories')
    
    # Common arguments
    parser.add_argument('--annotations_dir', type=str,
                        help='Directory containing YOLO annotation files')
    parser.add_argument('--species_mapping_file', type=str,
                        default='species_mapping.txt',
                        help='Species mapping file (default: species_mapping.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for organized dataset')
    
    # Processing parameters
    parser.add_argument('--buffer_ratio', type=float, default=0.1,
                        help='Buffer ratio around bounding boxes (default: 0.1)')
    parser.add_argument('--min_cutout_size', type=int, default=50,
                        help='Minimum cutout size in pixels (default: 50)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence for fish detection (default: 0.0)')
    
    # Organization parameters
    parser.add_argument('--labeled_families', type=str, nargs='*',
                        default=['Sparidae', 'Serranidae', 'Carangidae', 'Haemulidae'],
                        help='Fish families to include in labeled dataset')
    parser.add_argument('--max_per_species', type=int, default=None,
                        help='Maximum number of cutouts per species (default: no limit)')
    
    # Options
    parser.add_argument('--preview_only', action='store_true',
                        help='Only show statistics without processing files')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def load_species_mapping(mapping_file: str) -> Dict[int, Dict[str, str]]:
    """
    Load species mapping from file.
    
    Args:
        mapping_file: Path to species mapping file
        
    Returns:
        Dictionary mapping class_id to species info
    """
    mapping = {}
    
    if not os.path.exists(mapping_file):
        print(f"⚠️  Species mapping file not found: {mapping_file}")
        return mapping
    
    with open(mapping_file, 'r') as f:
        content = f.read()
    
    # Parse each line with format: "Family Species": ID,
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Match pattern: "Family Species": ID,
        match = re.match(r'"([^"]+)":\\s*(\\d+)', line)
        if match:
            species_full = match.group(1).strip()
            class_id = int(match.group(2))
            
            # Parse family and species
            if species_full and not species_full.isspace():
                parts = species_full.split()
                if len(parts) >= 2:
                    family = parts[0]
                    species = ' '.join(parts[1:])
                elif len(parts) == 1:
                    family = parts[0]
                    species = 'sp'
                else:
                    family = 'Unknown'
                    species = 'Unknown'
            else:
                family = 'Unknown'
                species = 'Unknown'
            
            mapping[class_id] = {
                'family': family,
                'species': species,
                'full_name': species_full
            }
    
    print(f"📋 Loaded {len(mapping)} species from mapping file")
    return mapping


def extract_fish_cutout(
    image_path: str, 
    annotation_line: str,
    buffer_ratio: float = 0.1,
    min_size: int = 50
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Extract fish cutout from YOLO annotation.
    
    Args:
        image_path: Path to image file
        annotation_line: YOLO format annotation line
        buffer_ratio: Extra padding around bounding box
        min_size: Minimum cutout size
        
    Returns:
        Tuple of (cutout_image, metadata) or None if failed
    """
    try:
        # Parse YOLO format: class_id x_center y_center width height [confidence]
        parts = annotation_line.strip().split()
        if len(parts) < 5:
            return None
            
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        img_height, img_width = image.shape[:2]
        
        # Convert YOLO to pixel coordinates
        x_center_px = int(x_center * img_width)
        y_center_px = int(y_center * img_height)
        width_px = int(width * img_width)
        height_px = int(height * img_height)
        
        # Add buffer
        buffer_w = int(width_px * buffer_ratio)
        buffer_h = int(height_px * buffer_ratio)
        
        # Calculate bounding box
        x1 = max(0, x_center_px - width_px // 2 - buffer_w)
        y1 = max(0, y_center_px - height_px // 2 - buffer_h)
        x2 = min(img_width, x_center_px + width_px // 2 + buffer_w)
        y2 = min(img_height, y_center_px + height_px // 2 + buffer_h)
        
        # Check minimum size
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
        
        # Extract cutout
        cutout = image[y1:y2, x1:x2]
        
        metadata = {
            'class_id': class_id,
            'confidence': confidence,
            'bbox': (x1, y1, x2, y2),
            'original_size': (img_width, img_height),
            'cutout_size': (x2 - x1, y2 - y1)
        }
        
        return cutout, metadata
        
    except Exception as e:
        print(f"❌ Error extracting cutout from {image_path}: {e}")
        return None


def process_single_directory(
    frames_dir: str,
    annotations_dir: str,
    species_mapping: Dict[int, Dict[str, str]],
    args
) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Process a single directory of frames and annotations.
    
    Args:
        frames_dir: Directory containing frame images
        annotations_dir: Directory containing annotation files
        species_mapping: Species ID mapping
        args: Command line arguments
        
    Returns:
        Dictionary of extracted cutouts organized by species
    """
    cutouts_by_species = defaultdict(list)
    
    # Get list of annotation files
    annotation_files = []
    if os.path.exists(annotations_dir):
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    
    print(f"📁 Processing {len(annotation_files)} annotation files...")
    
    for ann_file in annotation_files:
        # Find corresponding image
        base_name = os.path.splitext(ann_file)[0]
        
        # Try different image extensions
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_path = os.path.join(frames_dir, f"{base_name}{ext}")
            if os.path.exists(potential_path):
                image_file = potential_path
                break
        
        if image_file is None:
            if args.verbose:
                print(f"⚠️  No image found for {ann_file}")
            continue
        
        # Process annotations
        ann_path = os.path.join(annotations_dir, ann_file)
        with open(ann_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Extract cutout
                result = extract_fish_cutout(
                    image_file, line, 
                    buffer_ratio=args.buffer_ratio,
                    min_size=args.min_cutout_size
                )
                
                if result is None:
                    continue
                
                cutout, metadata = result
                
                # Check confidence threshold
                if metadata['confidence'] < args.confidence_threshold:
                    continue
                
                # Get species info
                class_id = metadata['class_id']
                if class_id in species_mapping:
                    species_info = species_mapping[class_id]
                    species_key = f"{species_info['family']}_{species_info['species']}"
                else:
                    species_key = f"unknown_{class_id}"
                
                # Generate unique filename
                filename = f"{base_name}_line{line_num}_{species_key}.jpg"
                
                # Store cutout info
                cutouts_by_species[species_key].append((filename, {
                    'cutout': cutout,
                    'metadata': metadata,
                    'species_info': species_mapping.get(class_id, {'family': 'Unknown', 'species': 'Unknown'})
                }))
    
    return cutouts_by_species


def organize_cutouts(
    cutouts_by_species: Dict[str, List[Tuple[str, Dict]]],
    output_dir: str,
    labeled_families: List[str],
    max_per_species: Optional[int] = None
):
    """
    Organize cutouts into labeled/unlabeled structure.
    
    Args:
        cutouts_by_species: Dictionary of cutouts by species
        output_dir: Output directory
        labeled_families: Families to include in labeled dataset
        max_per_species: Maximum cutouts per species
    """
    # Create output directories
    labeled_dir = os.path.join(output_dir, 'labeled')
    unlabeled_dir = os.path.join(output_dir, 'unlabeled')
    os.makedirs(labeled_dir, exist_ok=True)
    os.makedirs(unlabeled_dir, exist_ok=True)
    
    labeled_count = 0
    unlabeled_count = 0
    
    for species_key, cutout_list in cutouts_by_species.items():
        # Limit number per species if specified
        if max_per_species:
            cutout_list = cutout_list[:max_per_species]
        
        # Determine if this species should be labeled
        is_labeled = False
        family = cutout_list[0][1]['species_info'].get('family', 'Unknown')
        
        if family in labeled_families:
            is_labeled = True
            species_dir = os.path.join(labeled_dir, species_key)
            os.makedirs(species_dir, exist_ok=True)
        
        # Save cutouts
        for filename, data in cutout_list:
            cutout = data['cutout']
            
            if is_labeled:
                output_path = os.path.join(species_dir, filename)
                labeled_count += 1
            else:
                output_path = os.path.join(unlabeled_dir, filename)
                unlabeled_count += 1
            
            # Save image
            cv2.imwrite(output_path, cutout)
    
    print(f"✅ Organization complete:")
    print(f"  - Labeled cutouts: {labeled_count:,}")
    print(f"  - Unlabeled cutouts: {unlabeled_count:,}")
    print(f"  - Total species: {len(cutouts_by_species)}")


def process_multi_video_directories(
    multi_video_dir: str,
    species_mapping: Dict[int, Dict[str, str]],
    args
) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Process multiple video directories.
    
    Args:
        multi_video_dir: Directory containing video subdirectories
        species_mapping: Species ID mapping
        args: Command line arguments
        
    Returns:
        Combined dictionary of cutouts by species
    """
    all_cutouts = defaultdict(list)
    
    # Find video directories
    video_dirs = []
    for item in os.listdir(multi_video_dir):
        item_path = os.path.join(multi_video_dir, item)
        if os.path.isdir(item_path):
            video_dirs.append(item)
    
    print(f"📂 Found {len(video_dirs)} video directories")
    
    for video_dir in video_dirs:
        print(f"\\n🎬 Processing video directory: {video_dir}")
        
        video_path = os.path.join(multi_video_dir, video_dir)
        frames_dir = video_path  # Frames are directly in the video directory
        annotations_dir = video_path  # Annotations are also in the same directory
        
        # Process this video directory
        cutouts = process_single_directory(
            frames_dir, annotations_dir, species_mapping, args
        )
        
        # Merge results
        for species_key, cutout_list in cutouts.items():
            all_cutouts[species_key].extend(cutout_list)
        
        total_cutouts = sum(len(cutout_list) for cutout_list in cutouts.values())
        print(f"  📊 Extracted {total_cutouts:,} cutouts from {video_dir}")
    
    return all_cutouts


def main():
    """Main pipeline function."""
    args = parse_arguments()
    
    print("🐟 ViT-FishID Data Pipeline")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load species mapping
    species_mapping = load_species_mapping(args.species_mapping_file)
    
    if not species_mapping:
        print("⚠️  No species mapping loaded. All fish will be treated as 'unknown'.")
    
    # Process directories
    all_cutouts = defaultdict(list)
    
    if args.multi_video_dir:
        # Process multiple video directories
        all_cutouts = process_multi_video_directories(
            args.multi_video_dir, species_mapping, args
        )
    else:
        # Process single directory
        annotations_dir = args.annotations_dir or args.frames_dir
        all_cutouts = process_single_directory(
            args.frames_dir, annotations_dir, species_mapping, args
        )
    
    # Print statistics
    total_cutouts = sum(len(cutout_list) for cutout_list in all_cutouts.values())
    print(f"\\n📊 Extraction Summary:")
    print(f"  - Total cutouts: {total_cutouts:,}")
    print(f"  - Unique species: {len(all_cutouts)}")
    
    # Show species breakdown
    print(f"\\n🏷️  Species breakdown:")
    for species_key, cutout_list in sorted(all_cutouts.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(cutout_list)
        family = cutout_list[0][1]['species_info'].get('family', 'Unknown')
        labeled = '📌' if family in args.labeled_families else '🔍'
        print(f"  {labeled} {species_key}: {count:,} cutouts")
    
    if args.preview_only:
        print("\\n👀 Preview mode - no files saved")
        return
    
    # Organize cutouts
    print(f"\\n📂 Organizing cutouts...")
    organize_cutouts(
        all_cutouts, 
        args.output_dir, 
        args.labeled_families,
        args.max_per_species
    )
    
    # Create summary file
    summary = {
        'total_cutouts': total_cutouts,
        'unique_species': len(all_cutouts),
        'labeled_families': args.labeled_families,
        'species_counts': {k: len(v) for k, v in all_cutouts.items()},
        'parameters': {
            'buffer_ratio': args.buffer_ratio,
            'min_cutout_size': args.min_cutout_size,
            'confidence_threshold': args.confidence_threshold
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'extraction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\n🎉 Pipeline complete!")
    print(f"📄 Summary saved to: {summary_path}")
    print(f"🚀 Ready for training with: python train.py --data_dir {args.output_dir} --mode semi_supervised")


if __name__ == '__main__':
    main()
