#!/usr/bin/env python3
"""
🐟 Updated Multi-Video Fish Pipeline for ViT-FishID
Updated to work with a single consolidated species mapping file.

This handles:
- Multiple video directories with frames + annotations
- Single consolidated species mapping file
- Automatic cutout extraction and organization
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
    parser = argparse.ArgumentParser(description='Updated multi-video fish pipeline for ViT-FishID')
    
    parser.add_argument('--extracted_frames_dir', type=str, required=True,
                        help='Directory containing video subdirectories with frames and annotations')
    parser.add_argument('--species_mapping_file', type=str, required=True,
                        help='Single species mapping file (species_mapping.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for organized dataset')
    parser.add_argument('--buffer_ratio', type=float, default=0.1,
                        help='Buffer ratio around bounding boxes (default: 0.1)')
    parser.add_argument('--min_cutout_size', type=int, default=50,
                        help='Minimum cutout size in pixels (default: 50)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence for fish detection (default: 0.0)')
    parser.add_argument('--labeled_families', type=str, nargs='*', 
                        default=['Sparidae', 'Serranidae', 'Carangidae'],
                        help='Fish families to include in labeled dataset')
    parser.add_argument('--max_per_species', type=int, default=None,
                        help='Maximum number of cutouts per species (default: no limit)')
    parser.add_argument('--preview_only', action='store_true',
                        help='Only show statistics without processing files')
    
    return parser.parse_args()

def parse_consolidated_species_mapping(mapping_file: str) -> Dict[int, Dict[str, str]]:
    """Parse the consolidated species mapping file."""
    mapping = {}
    
    with open(mapping_file, 'r') as f:
        content = f.read()
    
    # Parse each line with format: "Family Species": ID,
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Match pattern: "Family Species": ID,
        match = re.match(r'"([^"]+)":\s*(\d+)', line)
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
    
    print(f"\n🎬 Processing video: {video_name}")
    
    # Find all frame/annotation pairs
    image_files = list(video_path.glob('*.jpg'))
    stats = defaultdict(int)
    
    for image_file in image_files:
        annotation_file = image_file.with_suffix('.txt')
        
        if not annotation_file.exists():
            continue
            
        # Read annotations
        with open(annotation_file, 'r') as f:
            annotation_lines = f.readlines()
        
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
            
            # Handle class ID 0 as unlabeled species
            if class_id == 0:
                category = 'unlabeled'
                output_subdir = Path(output_dir) / category
            elif class_id not in species_mapping:
                stats['unknown_species'] += 1
                continue
            else:
                species_info = species_mapping[class_id]
                family = species_info['family']
                species = species_info['species']
                
                # Determine if labeled or unlabeled based on family
                is_labeled = family in args.labeled_families
                category = 'labeled' if is_labeled else 'unlabeled'
                
                # Create output directory structure
                if is_labeled:
                    # Use family_species format for labeled data
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
            
            # Update family statistics (only for known species)
            if class_id != 0 and class_id in species_mapping:
                family = species_mapping[class_id]['family']
                stats[f'{category}_{family}'] += 1
            
    return dict(stats)

def analyze_dataset(species_mapping: Dict[int, Dict[str, str]]) -> Dict:
    """Analyze the dataset to show statistics."""
    analysis = {
        'total_species': len(species_mapping),
        'families': set(),
        'species_per_family': defaultdict(list),
        'class_distribution': {}
    }
    
    for class_id, species_info in species_mapping.items():
        family = species_info['family']
        species = species_info['species']
        full_name = species_info['full_name']
        
        analysis['families'].add(family)
        analysis['species_per_family'][family].append(species)
        analysis['class_distribution'][full_name] = 0  # Will be updated during processing
    
    analysis['families'] = sorted(list(analysis['families']))
    
    return analysis

def print_dataset_analysis(analysis: Dict, labeled_families: List[str]):
    """Print comprehensive dataset analysis."""
    print("\n" + "="*60)
    print("🐟 DATASET ANALYSIS")
    print("="*60)
    
    print(f"📊 Overview:")
    print(f"   • Total unique species: {analysis['total_species']}")
    print(f"   • Total families: {len(analysis['families'])}")
    
    print(f"\n🏷️ Labeled families (will be in labeled dataset):")
    for family in labeled_families:
        if family in analysis['families']:
            species_count = len(analysis['species_per_family'][family])
            print(f"   ✅ {family}: {species_count} species")
        else:
            print(f"   ❌ {family}: Not found in dataset")
    
    print(f"\n🔍 Unlabeled families (will be in unlabeled dataset):")
    unlabeled_families = [f for f in analysis['families'] if f not in labeled_families]
    for family in unlabeled_families[:10]:  # Show first 10
        species_count = len(analysis['species_per_family'][family])
        print(f"   • {family}: {species_count} species")
    if len(unlabeled_families) > 10:
        print(f"   ... and {len(unlabeled_families) - 10} more families")

def main():
    """Main processing function."""
    args = parse_arguments()
    
    print("🐟 Updated Multi-Video Fish Pipeline for ViT-FishID")
    print("=" * 60)
    
    # Validate input directories
    if not os.path.exists(args.extracted_frames_dir):
        print(f"❌ Extracted frames directory not found: {args.extracted_frames_dir}")
        return
        
    if not os.path.exists(args.species_mapping_file):
        print(f"❌ Species mapping file not found: {args.species_mapping_file}")
        return
    
    # Load consolidated species mapping
    print(f"\n📖 Loading species mapping from {args.species_mapping_file}...")
    species_mapping = parse_consolidated_species_mapping(args.species_mapping_file)
    
    if not species_mapping:
        print("❌ No species mappings found in file!")
        return
    
    print(f"✅ Loaded {len(species_mapping)} species mappings")
    
    # Analyze dataset
    analysis = analyze_dataset(species_mapping)
    print_dataset_analysis(analysis, args.labeled_families)
    
    if args.preview_only:
        print(f"\n👀 Preview mode - no files will be processed")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each video directory
    extracted_frames_path = Path(args.extracted_frames_dir)
    all_video_dirs = [d for d in extracted_frames_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n🚀 Processing {len(all_video_dirs)} video directories...")
    total_stats = defaultdict(int)
    processed_videos = 0
    
    for video_dir in all_video_dirs:
        video_name = video_dir.name
        
        # Process this video with the consolidated mapping
        video_stats = process_video_directory(str(video_dir), species_mapping, 
                                            args.output_dir, args)
        
        # Accumulate statistics
        for key, value in video_stats.items():
            total_stats[key] += value
            
        processed_videos += 1
        
        # Show progress
        if processed_videos % 5 == 0:
            print(f"   📊 Processed {processed_videos} videos so far...")
    
    # Print final statistics
    print(f"\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)
    print(f"📊 Final Statistics:")
    print(f"   • Videos processed: {processed_videos}")
    print(f"   • Total cutouts extracted: {total_stats['total_cutouts']:,}")
    print(f"   • Labeled cutouts: {total_stats['labeled']:,}")
    print(f"   • Unlabeled cutouts: {total_stats['unlabeled']:,}")
    print(f"   • Failed extractions: {total_stats['failed_extractions']:,}")
    print(f"   • Too small: {total_stats['too_small']:,}")
    print(f"   • Low confidence: {total_stats['low_confidence']:,}")
    print(f"   • Unknown species: {total_stats['unknown_species']:,}")
    
    # Show labeled species breakdown
    print(f"\n📚 Labeled species breakdown:")
    for family in args.labeled_families:
        count = total_stats.get(f'labeled_{family}', 0)
        if count > 0:
            print(f"   • {family}: {count:,} cutouts")
    
    # Save dataset info
    dataset_info = {
        'total_cutouts': total_stats['total_cutouts'],
        'labeled_cutouts': total_stats['labeled'],
        'unlabeled_cutouts': total_stats['unlabeled'],
        'labeled_families': args.labeled_families,
        'processing_args': vars(args),
        'videos_processed': processed_videos,
        'total_species_in_mapping': len(species_mapping)
    }
    
    info_file = output_path / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n💾 Dataset info saved to: {info_file}")
    print(f"📁 Organized dataset ready at: {args.output_dir}")
    print(f"\n🚀 Ready for training!")
    print(f"Run: python main_semi_supervised.py --data_dir {args.output_dir}")

if __name__ == '__main__':
    main()
