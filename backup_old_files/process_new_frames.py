#!/usr/bin/env python3
"""
🐟 Process New 21-05_W Frames for ViT-FishID
Extract cutouts from new 21-05_W frame directories and categorize them 
into labeled/unlabeled based on existing species mapping.
"""

import os
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import shutil
from collections import defaultdict, Counter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process new 21-05_W frames for ViT-FishID')
    
    parser.add_argument('--frames_dir', type=str, 
                        default='/Users/catalinathomson/Desktop/Fish/ViT-FishID/Frames/extracted_frames',
                        help='Directory containing frame subdirectories')
    parser.add_argument('--species_mapping_file', type=str,
                        default='/Users/catalinathomson/Desktop/Fish/ViT-FishID/Frames/species_mapping.txt',
                        help='Species mapping file')
    parser.add_argument('--output_dir', type=str, 
                        default='/Users/catalinathomson/Desktop/Fish/ViT-FishID/fish_cutouts',
                        help='Output directory (existing fish_cutouts)')
    parser.add_argument('--buffer_ratio', type=float, default=0.1,
                        help='Buffer ratio around bounding boxes (default: 0.1)')
    parser.add_argument('--min_cutout_size', type=int, default=50,
                        help='Minimum cutout size in pixels (default: 50)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence for fish detection (default: 0.0)')
    parser.add_argument('--labeled_families', type=str, nargs='*', 
                        default=['Sparidae', 'Serranidae', 'Carangidae'],
                        help='Fish families to include in labeled dataset')
    parser.add_argument('--preview_only', action='store_true',
                        help='Only show statistics without processing files')
    
    return parser.parse_args()

def parse_species_mapping_file(mapping_file: str) -> Dict[int, Dict[str, str]]:
    """Parse the species mapping file to extract class ID -> species info."""
    mapping = {}
    
    with open(mapping_file, 'r') as f:
        content = f.read()
    
    # Parse species entries from the JSON-like format
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Match pattern: "Family Species": ID,
        match = re.match(r'"([^"]+)":\s*(\d+),?', line)
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
    
    print(f"📖 Loaded {len(mapping)} species from mapping file")
    return mapping

def extract_cutout_from_annotation(image_path: str, annotation_line: str, 
                                 buffer_ratio: float = 0.1) -> Optional[Tuple[any, Dict]]:
    """Extract fish cutout from YOLO annotation line."""
    try:
        # Parse YOLO format: class_id x_center y_center width height confidence
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
        
        # Convert YOLO coordinates to pixel coordinates
        x_center_px = int(x_center * img_width)
        y_center_px = int(y_center * img_height)
        width_px = int(width * img_width)
        height_px = int(height * img_height)
        
        # Add buffer
        buffer_w = int(width_px * buffer_ratio)
        buffer_h = int(height_px * buffer_ratio)
        
        # Calculate bounding box with buffer
        x1 = max(0, x_center_px - (width_px // 2) - buffer_w)
        y1 = max(0, y_center_px - (height_px // 2) - buffer_h)
        x2 = min(img_width, x_center_px + (width_px // 2) + buffer_w)
        y2 = min(img_height, y_center_px + (height_px // 2) + buffer_h)
        
        # Extract cutout
        cutout = image[y1:y2, x1:x2]
        
        if cutout.size == 0:
            return None
            
        cutout_info = {
            'class_id': class_id,
            'confidence': confidence,
            'original_size': (img_width, img_height),
            'cutout_size': (x2-x1, y2-y1),
            'bbox': (x1, y1, x2, y2)
        }
        
        return cutout, cutout_info
        
    except Exception as e:
        print(f"❌ Error extracting cutout: {e}")
        return None

def process_video_directory(video_dir: str, species_mapping: Dict[int, Dict[str, str]], 
                          output_dir: str, args) -> Dict[str, int]:
    """Process a single video directory."""
    video_path = Path(video_dir)
    video_name = video_path.name
    
    print(f"🎬 Processing: {video_name}")
    
    # Find all frame/annotation pairs
    image_files = list(video_path.glob('*.jpg'))
    stats = defaultdict(int)
    
    for image_file in image_files:
        annotation_file = image_file.with_suffix('.txt')
        
        if not annotation_file.exists():
            stats['no_annotation'] += 1
            continue
            
        # Read annotations
        try:
            with open(annotation_file, 'r') as f:
                annotation_lines = f.readlines()
        except Exception as e:
            print(f"⚠️ Error reading {annotation_file}: {e}")
            stats['read_error'] += 1
            continue
        
        for i, line in enumerate(annotation_lines):
            line = line.strip()
            if not line:
                continue
                
            # Extract cutout
            result = extract_cutout_from_annotation(str(image_file), line, args.buffer_ratio)
            if result is None:
                stats['failed_extractions'] += 1
                continue
                
            cutout, cutout_info = result
            
            # Check minimum size
            cutout_h, cutout_w = cutout.shape[:2]
            if cutout_w < args.min_cutout_size or cutout_h < args.min_cutout_size:
                stats['too_small'] += 1
                continue
                
            # Check confidence
            if cutout_info['confidence'] < args.confidence_threshold:
                stats['low_confidence'] += 1
                continue
                
            # Get species information
            class_id = cutout_info['class_id']
            if class_id not in species_mapping:
                stats['unknown_species'] += 1
                continue
                
            species_info = species_mapping[class_id]
            family = species_info['family']
            species = species_info['species']
            
            # Determine if labeled or unlabeled
            is_labeled = family in args.labeled_families
            category = 'labeled' if is_labeled else 'unlabeled'
            
            # Create output directory structure
            if is_labeled:
                # Use family_species format for labeled data (match existing structure)
                species_name = f"{family}_{species.replace(' ', '_')}"
                output_subdir = Path(output_dir) / category / species_name
            else:
                output_subdir = Path(output_dir) / category
                
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            frame_name = image_file.stem
            cutout_filename = f"{video_name}_{frame_name}_det{i:02d}_cls{class_id}.jpg"
            output_path = output_subdir / cutout_filename
            
            # Save cutout (if not preview mode)
            if not args.preview_only:
                cv2.imwrite(str(output_path), cutout)
            
            # Update statistics
            stats['total_cutouts'] += 1
            stats[category] += 1
            stats[f'{category}_{family}'] += 1
            
    return dict(stats)

def main():
    """Main processing function."""
    args = parse_arguments()
    
    print("🐟 Processing New 21-05_W Frames for ViT-FishID")
    print("=" * 60)
    
    # Validate input files/directories
    if not os.path.exists(args.frames_dir):
        print(f"❌ Frames directory not found: {args.frames_dir}")
        return
        
    if not os.path.exists(args.species_mapping_file):
        print(f"❌ Species mapping file not found: {args.species_mapping_file}")
        return
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory not found: {args.output_dir}")
        return
    
    # Load species mapping
    print(f"📖 Loading species mapping from {args.species_mapping_file}...")
    species_mapping = parse_species_mapping_file(args.species_mapping_file)
    
    if not species_mapping:
        print("❌ No species mapping loaded!")
        return
    
    # Find all 21-05_W directories
    frames_path = Path(args.frames_dir)
    w_directories = [d for d in frames_path.iterdir() 
                     if d.is_dir() and d.name.startswith('21-05_W')]
    
    print(f"\\n🔍 Found {len(w_directories)} directories starting with '21-05_W':")
    for d in w_directories[:10]:  # Show first 10
        print(f"   📁 {d.name}")
    if len(w_directories) > 10:
        print(f"   ... and {len(w_directories) - 10} more directories")
    
    if args.preview_only:
        print(f"\\n👀 Preview mode - no files will be processed")
        
        # Quick preview of what would be processed
        total_frames = 0
        total_annotations = 0
        for video_dir in w_directories[:5]:  # Sample first 5
            image_files = list(video_dir.glob('*.jpg'))
            annotation_files = list(video_dir.glob('*.txt'))
            total_frames += len(image_files)
            total_annotations += len(annotation_files)
            print(f"   📊 {video_dir.name}: {len(image_files)} frames, {len(annotation_files)} annotations")
        
        print(f"\\n📈 Sample totals (first 5 directories): {total_frames} frames, {total_annotations} annotations")
        return
    
    # Process each 21-05_W directory
    print(f"\\n🚀 Processing {len(w_directories)} directories...")
    total_stats = defaultdict(int)
    processed_dirs = 0
    
    for video_dir in w_directories:
        # Process this directory
        video_stats = process_video_directory(str(video_dir), species_mapping, 
                                            args.output_dir, args)
        
        # Accumulate statistics
        for key, value in video_stats.items():
            total_stats[key] += value
            
        processed_dirs += 1
        
        # Show progress every 10 directories
        if processed_dirs % 10 == 0:
            print(f"   📊 Processed {processed_dirs}/{len(w_directories)} directories...")
    
    # Print final statistics
    print(f"\\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)
    print(f"📊 Final Statistics:")
    print(f"   • Directories processed: {processed_dirs}")
    print(f"   • Total cutouts extracted: {total_stats['total_cutouts']:,}")
    print(f"   • Labeled cutouts: {total_stats['labeled']:,}")
    print(f"   • Unlabeled cutouts: {total_stats['unlabeled']:,}")
    print(f"   • Failed extractions: {total_stats['failed_extractions']:,}")
    print(f"   • Too small: {total_stats['too_small']:,}")
    print(f"   • Low confidence: {total_stats['low_confidence']:,}")
    print(f"   • Unknown species: {total_stats['unknown_species']:,}")
    print(f"   • No annotations: {total_stats['no_annotation']:,}")
    
    # Show labeled species breakdown
    print(f"\\n📚 Labeled species breakdown:")
    for family in args.labeled_families:
        count = total_stats.get(f'labeled_{family}', 0)
        if count > 0:
            print(f"   • {family}: {count:,} cutouts")
    
    # Update existing dataset info
    dataset_info_file = Path(args.output_dir) / 'dataset_info.json'
    if dataset_info_file.exists():
        with open(dataset_info_file, 'r') as f:
            dataset_info = json.load(f)
        
        # Update counts
        dataset_info['total_cutouts'] += total_stats['total_cutouts']
        dataset_info['labeled_cutouts'] += total_stats['labeled']
        dataset_info['unlabeled_cutouts'] += total_stats['unlabeled']
        
        # Add processing info
        if 'processing_history' not in dataset_info:
            dataset_info['processing_history'] = []
        
        dataset_info['processing_history'].append({
            'date': '2025-08-06',
            'source': '21-05_W frames',
            'directories_processed': processed_dirs,
            'cutouts_added': total_stats['total_cutouts'],
            'labeled_added': total_stats['labeled'],
            'unlabeled_added': total_stats['unlabeled']
        })
        
        with open(dataset_info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\\n💾 Updated dataset info: {dataset_info_file}")
    
    print(f"\\n🎉 New cutouts added to existing fish_cutouts dataset!")
    print(f"📁 Dataset location: {args.output_dir}")
    print(f"\\n🚀 Ready for training with expanded dataset!")

if __name__ == '__main__':
    main()
