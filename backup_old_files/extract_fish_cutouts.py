#!/usr/bin/env python3
"""
Extract fish cutouts from bounding box annotations for ViT training.

This script processes images with YOLO-format bounding box annotations,
extracts fish cutouts with a buffer, and organizes them for semi-supervised learning.

Usage:
    python extract_fish_cutouts.py --frames_dir /path/to/frames --annotations_dir /path/to/annotations --output_dir /path/to/cutouts

Author: GitHub Copilot
Date: 2025
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import shutil
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract fish cutouts from bounding box annotations')
    
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing frame images')
    parser.add_argument('--annotations_dir', type=str, required=True,
                        help='Directory containing YOLO annotation txt files')
    parser.add_argument('--species_mapping_file', type=str, default=None,
                        help='JSON file with species ID to name mapping (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for extracted fish cutouts')
    parser.add_argument('--buffer_ratio', type=float, default=0.1,
                        help='Buffer around bounding box as ratio of box size (default: 0.1)')
    parser.add_argument('--min_size', type=int, default=32,
                        help='Minimum size for fish cutouts in pixels (default: 32)')
    parser.add_argument('--max_size', type=int, default=512,
                        help='Maximum size for fish cutouts in pixels (default: 512)')
    parser.add_argument('--image_quality', type=int, default=95,
                        help='JPEG quality for saved cutouts (default: 95)')
    parser.add_argument('--create_species_mapping', action='store_true',
                        help='Create species mapping from full_final_frames directory')
    parser.add_argument('--full_final_frames_dir', type=str, default=None,
                        help='Directory to scan for species names (for creating mapping)')
    
    return parser.parse_args()


def create_species_mapping_from_directory(full_final_frames_dir: str) -> Dict[int, str]:
    """
    Create species mapping by analyzing the full_final_frames directory.
    Assumes directory contains species-named subdirectories or files.
    """
    if not os.path.exists(full_final_frames_dir):
        print(f"⚠️ Directory not found: {full_final_frames_dir}")
        return {}
    
    species_names = set()
    path = Path(full_final_frames_dir)
    
    # Look for subdirectories (species folders)
    for item in path.iterdir():
        if item.is_dir():
            species_names.add(item.name.lower())
    
    # If no subdirectories, look for files with species names
    if not species_names:
        for item in path.rglob('*'):
            if item.is_file() and item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                # Extract potential species name from filename
                filename = item.stem.lower()
                # Split on common separators
                parts = filename.replace('_', ' ').replace('-', ' ').split()
                for part in parts:
                    if len(part) > 3 and part.isalpha():  # Likely species name
                        species_names.add(part)
    
    # Create mapping (species_id -> species_name)
    species_list = sorted(list(species_names))
    species_mapping = {i: species for i, species in enumerate(species_list)}
    
    print(f"🐟 Found {len(species_mapping)} species:")
    for species_id, species_name in species_mapping.items():
        print(f"  {species_id}: {species_name}")
    
    return species_mapping


def load_species_mapping(mapping_file: str) -> Dict[int, str]:
    """Load species mapping from JSON file."""
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Convert string keys to int if needed
        if isinstance(list(mapping.keys())[0], str):
            mapping = {int(k): v for k, v in mapping.items()}
        
        print(f"📖 Loaded species mapping from {mapping_file}")
        for species_id, species_name in mapping.items():
            print(f"  {species_id}: {species_name}")
        
        return mapping
    
    except Exception as e:
        print(f"❌ Error loading species mapping: {e}")
        return {}


def save_species_mapping(mapping: Dict[int, str], output_file: str):
    """Save species mapping to JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"💾 Species mapping saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving species mapping: {e}")


def parse_yolo_annotation(annotation_file: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse YOLO annotation file.
    
    Returns:
        List of tuples: (class_id, x_center, y_center, width, height)
        All coordinates are normalized (0-1)
    """
    annotations = []
    
    try:
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append((class_id, x_center, y_center, width, height))
    
    except Exception as e:
        print(f"⚠️ Error parsing {annotation_file}: {e}")
    
    return annotations


def extract_fish_cutout(
    image: np.ndarray, 
    bbox: Tuple[float, float, float, float], 
    buffer_ratio: float = 0.1,
    min_size: int = 32,
    max_size: int = 512
) -> Optional[np.ndarray]:
    """
    Extract fish cutout from image using bounding box.
    
    Args:
        image: Input image (H, W, C)
        bbox: YOLO format (x_center, y_center, width, height) normalized
        buffer_ratio: Buffer around bounding box as ratio of box size
        min_size: Minimum cutout size in pixels
        max_size: Maximum cutout size in pixels
    
    Returns:
        Extracted fish cutout or None if invalid
    """
    h, w = image.shape[:2]
    x_center, y_center, box_width, box_height = bbox
    
    # Convert from normalized to pixel coordinates
    x_center_px = x_center * w
    y_center_px = y_center * h
    box_width_px = box_width * w
    box_height_px = box_height * h
    
    # Add buffer
    buffer_w = box_width_px * buffer_ratio
    buffer_h = box_height_px * buffer_ratio
    
    # Calculate bounding box with buffer
    x1 = int(x_center_px - (box_width_px + buffer_w) / 2)
    y1 = int(y_center_px - (box_height_px + buffer_h) / 2)
    x2 = int(x_center_px + (box_width_px + buffer_w) / 2)
    y2 = int(y_center_px + (box_height_px + buffer_h) / 2)
    
    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Check minimum size
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    
    # Extract cutout
    cutout = image[y1:y2, x1:x2]
    
    # Resize if too large
    cutout_h, cutout_w = cutout.shape[:2]
    if max(cutout_h, cutout_w) > max_size:
        scale = max_size / max(cutout_h, cutout_w)
        new_w = int(cutout_w * scale)
        new_h = int(cutout_h * scale)
        cutout = cv2.resize(cutout, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return cutout


def process_frame_annotations(
    frame_path: str,
    annotation_path: str,
    species_mapping: Dict[int, str],
    output_dir: str,
    args
) -> Dict[str, int]:
    """
    Process a single frame and its annotations.
    
    Returns:
        Dictionary with extraction statistics
    """
    stats = {'total_detections': 0, 'successful_extractions': 0, 'failed_extractions': 0}
    
    # Load image
    image = cv2.imread(frame_path)
    if image is None:
        print(f"⚠️ Could not load image: {frame_path}")
        return stats
    
    # Parse annotations
    annotations = parse_yolo_annotation(annotation_path)
    stats['total_detections'] = len(annotations)
    
    if not annotations:
        return stats
    
    # Extract cutouts for each detection
    frame_name = Path(frame_path).stem
    
    for i, (class_id, x_center, y_center, width, height) in enumerate(annotations):
        # Get species name
        species_name = species_mapping.get(class_id, f"unknown_species_{class_id}")
        
        # Create species directory
        species_dir = Path(output_dir) / species_name
        species_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract cutout
        cutout = extract_fish_cutout(
            image, 
            (x_center, y_center, width, height),
            buffer_ratio=args.buffer_ratio,
            min_size=args.min_size,
            max_size=args.max_size
        )
        
        if cutout is not None:
            # Save cutout
            cutout_filename = f"{frame_name}_{species_name}_{i:03d}.jpg"
            cutout_path = species_dir / cutout_filename
            
            # Save with specified quality
            cv2.imwrite(
                str(cutout_path), 
                cutout, 
                [cv2.IMWRITE_JPEG_QUALITY, args.image_quality]
            )
            
            stats['successful_extractions'] += 1
        else:
            stats['failed_extractions'] += 1
    
    return stats


def find_matching_files(frames_dir: str, annotations_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching frame and annotation files.
    
    Returns:
        List of tuples: (frame_path, annotation_path)
    """
    frames_path = Path(frames_dir)
    annotations_path = Path(annotations_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    frame_files = {}
    
    for frame_file in frames_path.rglob('*'):
        if frame_file.suffix.lower() in image_extensions:
            stem = frame_file.stem
            frame_files[stem] = str(frame_file)
    
    # Find matching annotation files
    matching_pairs = []
    
    for annotation_file in annotations_path.rglob('*.txt'):
        stem = annotation_file.stem
        if stem in frame_files:
            matching_pairs.append((frame_files[stem], str(annotation_file)))
    
    print(f"📊 Found {len(matching_pairs)} matching frame-annotation pairs")
    return matching_pairs


def main():
    """Main extraction function."""
    args = parse_arguments()
    
    print("🐟 Fish Cutout Extractor")
    print("=" * 40)
    print(f"Frames directory: {args.frames_dir}")
    print(f"Annotations directory: {args.annotations_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Buffer ratio: {args.buffer_ratio}")
    
    # Validate input directories
    if not os.path.exists(args.frames_dir):
        print(f"❌ Frames directory does not exist: {args.frames_dir}")
        return
    
    if not os.path.exists(args.annotations_dir):
        print(f"❌ Annotations directory does not exist: {args.annotations_dir}")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Handle species mapping
    species_mapping = {}
    
    if args.create_species_mapping and args.full_final_frames_dir:
        print(f"\n🔍 Creating species mapping from: {args.full_final_frames_dir}")
        species_mapping = create_species_mapping_from_directory(args.full_final_frames_dir)
        
        # Save the mapping
        mapping_file = Path(args.output_dir) / "species_mapping.json"
        save_species_mapping(species_mapping, str(mapping_file))
        
    elif args.species_mapping_file:
        print(f"\n📖 Loading species mapping from: {args.species_mapping_file}")
        species_mapping = load_species_mapping(args.species_mapping_file)
    
    else:
        print("\n⚠️ No species mapping provided. Using class IDs as species names.")
        # We'll create mapping on-the-fly based on encountered class IDs
    
    # Find matching frame-annotation pairs
    print(f"\n🔍 Scanning for matching files...")
    matching_pairs = find_matching_files(args.frames_dir, args.annotations_dir)
    
    if not matching_pairs:
        print("❌ No matching frame-annotation pairs found!")
        return
    
    # Process all frames
    print(f"\n🚀 Processing {len(matching_pairs)} frames...")
    
    total_stats = {
        'processed_frames': 0,
        'total_detections': 0,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'species_counts': {}
    }
    
    progress_bar = tqdm(matching_pairs, desc="Processing frames")
    
    for frame_path, annotation_path in progress_bar:
        frame_stats = process_frame_annotations(
            frame_path, annotation_path, species_mapping, args.output_dir, args
        )
        
        # Update total statistics
        total_stats['processed_frames'] += 1
        total_stats['total_detections'] += frame_stats['total_detections']
        total_stats['successful_extractions'] += frame_stats['successful_extractions']
        total_stats['failed_extractions'] += frame_stats['failed_extractions']
        
        # Update progress bar
        progress_bar.set_postfix({
            'Detections': total_stats['total_detections'],
            'Extracted': total_stats['successful_extractions'],
            'Failed': total_stats['failed_extractions']
        })
    
    # Count species
    output_path = Path(args.output_dir)
    for species_dir in output_path.iterdir():
        if species_dir.is_dir():
            count = len(list(species_dir.glob('*.jpg')))
            total_stats['species_counts'][species_dir.name] = count
    
    # Print final results
    print(f"\n✅ Extraction completed!")
    print(f"📊 Results:")
    print(f"  - Processed frames: {total_stats['processed_frames']:,}")
    print(f"  - Total detections: {total_stats['total_detections']:,}")
    print(f"  - Successful extractions: {total_stats['successful_extractions']:,}")
    print(f"  - Failed extractions: {total_stats['failed_extractions']:,}")
    
    print(f"\n🐟 Species distribution:")
    for species, count in total_stats['species_counts'].items():
        print(f"  - {species}: {count:,} cutouts")
    
    # Save extraction report
    report_path = output_path / "extraction_report.txt"
    with open(report_path, 'w') as f:
        f.write("🐟 Fish Cutout Extraction Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Processed frames: {total_stats['processed_frames']:,}\n")
        f.write(f"Total detections: {total_stats['total_detections']:,}\n")
        f.write(f"Successful extractions: {total_stats['successful_extractions']:,}\n")
        f.write(f"Failed extractions: {total_stats['failed_extractions']:,}\n\n")
        
        f.write("Species Distribution:\n")
        f.write("-" * 20 + "\n")
        for species, count in total_stats['species_counts'].items():
            f.write(f"{species}: {count:,} cutouts\n")
        
        f.write(f"\nExtraction Settings:\n")
        f.write(f"Buffer ratio: {args.buffer_ratio}\n")
        f.write(f"Min size: {args.min_size}px\n")
        f.write(f"Max size: {args.max_size}px\n")
        f.write(f"JPEG quality: {args.image_quality}\n")
    
    print(f"\n📄 Extraction report saved to: {report_path}")
    
    # Suggest next steps
    print(f"\n🚀 Next Steps:")
    print(f"1. Review extracted cutouts in: {args.output_dir}")
    print(f"2. Organize for semi-supervised learning:")
    print(f"   python organize_fish_data.py --input_dir {args.output_dir} --output_dir /path/to/organized/dataset")
    print(f"3. Train ViT model:")
    print(f"   python main_semi_supervised.py --data_dir /path/to/organized/dataset")


if __name__ == '__main__':
    main()
