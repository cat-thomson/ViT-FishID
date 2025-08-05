#!/usr/bin/env python3
"""
🐟 Multi-Video Fish Pipeline for ViT-FishID
Specialized pipeline for processing fish data organized by video directories
with existing species class mappings.

This handles:
- Multiple video directories with frames + annotations
- Species-to-class-ID mapping files
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
    parser = argparse.ArgumentParser(description='Multi-video fish pipeline for ViT-FishID')
    
    parser.add_argument('--extracted_frames_dir', type=str, required=True,
                        help='Directory containing video subdirectories with frames and annotations')
    parser.add_argument('--species_mappings_dir', type=str, required=True,
                        help='Directory containing species_class_mapping_*.txt files')
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

def parse_species_mapping_file(mapping_file: str) -> Dict[int, Dict[str, str]]:
    """Parse a species mapping file to extract class ID -> species info."""
    mapping = {}
    
    with open(mapping_file, 'r') as f:
        content = f.read()
    
    # Extract directory name from the file
    directory_match = re.search(r'directories matching: (.+)', content)
    if directory_match:
        directory_name = directory_match.group(1)
    else:
        # Fall back to filename
        directory_name = Path(mapping_file).stem.replace('species_class_mapping_', '')
    
    # Parse species entries
    for line in content.split('\n'):
        # Match pattern: "ID: Family Species (count: N)"
        match = re.match(r'(\d+):\s*(.+?)\s*\(count:\s*(\d+)\)', line)
        if match:
            class_id = int(match.group(1))
            species_info = match.group(2).strip()
            count = int(match.group(3))
            
            # Parse family and species
            if species_info and not species_info.isspace():
                parts = species_info.split()
                if len(parts) >= 2:
                    family = parts[0]
                    species = ' '.join(parts[1:])
                else:
                    family = 'Unknown'
                    species = species_info if species_info else 'Unknown'
            else:
                family = 'Unknown'
                species = 'Unknown'
            
            mapping[class_id] = {
                'family': family,
                'species': species,
                'count': count,
                'directory': directory_name
            }
    
    return mapping

def load_all_species_mappings(mappings_dir: str) -> Dict[str, Dict[int, Dict[str, str]]]:
    """Load all species mapping files and organize by directory name."""
    all_mappings = {}
    
    mappings_path = Path(mappings_dir)
    for mapping_file in mappings_path.glob('species_class_mapping_*.txt'):
        # Extract directory name from filename
        directory_name = mapping_file.stem.replace('species_class_mapping_', '')
        
        mapping = parse_species_mapping_file(str(mapping_file))
        if mapping:
            all_mappings[directory_name] = mapping
            print(f"📁 Loaded mapping for {directory_name}: {len(mapping)} species")
    
    return all_mappings

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
            stats[f'{category}_{family}'] += 1
            
    return dict(stats)

def analyze_datasets(all_mappings: Dict[str, Dict[int, Dict[str, str]]]) -> Dict:
    """Analyze the dataset to show statistics."""
    analysis = {
        'total_videos': len(all_mappings),
        'total_species': 0,
        'families': set(),
        'species_per_family': defaultdict(list),
        'class_distribution': defaultdict(int)
    }
    
    all_species = set()
    
    for video_name, mapping in all_mappings.items():
        for class_id, species_info in mapping.items():
            family = species_info['family']
            species = species_info['species']
            count = species_info['count']
            
            species_key = f"{family} {species}"
            all_species.add(species_key)
            analysis['families'].add(family)
            analysis['species_per_family'][family].append(species)
            analysis['class_distribution'][species_key] += count
    
    analysis['total_species'] = len(all_species)
    analysis['families'] = sorted(list(analysis['families']))
    
    return analysis

def print_dataset_analysis(analysis: Dict, labeled_families: List[str]):
    """Print comprehensive dataset analysis."""
    print("\n" + "="*60)
    print("🐟 DATASET ANALYSIS")
    print("="*60)
    
    print(f"📊 Overview:")
    print(f"   • Total videos: {analysis['total_videos']}")
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
    
    print(f"\n📈 Top species by detection count:")
    sorted_species = sorted(analysis['class_distribution'].items(), 
                           key=lambda x: x[1], reverse=True)
    for species, count in sorted_species[:15]:
        family = species.split()[0] if species.split() else 'Unknown'
        label_status = "📚" if family in labeled_families else "🔄"
        print(f"   {label_status} {species}: {count} detections")

def main():
    """Main processing function."""
    args = parse_arguments()
    
    print("🐟 Multi-Video Fish Pipeline for ViT-FishID")
    print("=" * 60)
    
    # Validate input directories
    if not os.path.exists(args.extracted_frames_dir):
        print(f"❌ Extracted frames directory not found: {args.extracted_frames_dir}")
        return
        
    if not os.path.exists(args.species_mappings_dir):
        print(f"❌ Species mappings directory not found: {args.species_mappings_dir}")
        return
    
    # Load all species mappings
    print(f"\n📖 Loading species mappings from {args.species_mappings_dir}...")
    all_mappings = load_all_species_mappings(args.species_mappings_dir)
    
    if not all_mappings:
        print("❌ No species mapping files found!")
        return
    
    # Analyze dataset
    analysis = analyze_datasets(all_mappings)
    print_dataset_analysis(analysis, args.labeled_families)
    
    if args.preview_only:
        print(f"\n👀 Preview mode - no files will be processed")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each video directory
    print(f"\n🚀 Processing {len(all_mappings)} videos...")
    total_stats = defaultdict(int)
    
    extracted_frames_path = Path(args.extracted_frames_dir)
    processed_videos = 0
    
    for video_dir in extracted_frames_path.iterdir():
        if not video_dir.is_dir() or video_dir.name.startswith('.'):
            continue
            
        video_name = video_dir.name
        
        # Find corresponding species mapping
        species_mapping = None
        for mapping_key, mapping in all_mappings.items():
            if mapping_key in video_name or video_name in mapping_key:
                species_mapping = mapping
                break
        
        if species_mapping is None:
            print(f"⚠️  No species mapping found for video: {video_name}")
            continue
        
        # Process this video
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
        'videos_processed': processed_videos
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
