#!/usr/bin/env python3
"""Simple test to verify argument parsing for train/test/validation splits."""

import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ViT-FishID Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--mode', type=str, choices=['supervised', 'semi_supervised'], 
                        default='semi_supervised',
                        help='Training mode: supervised or semi_supervised')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test split ratio (default: 0.2)')
    
    return parser

if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args(['--data_dir', './test'])  # Test with minimal args
    
    print("✅ Successfully parsed arguments:")
    print(f"  - data_dir: {args.data_dir}")
    print(f"  - mode: {args.mode}")
    print(f"  - val_split: {args.val_split}")
    print(f"  - test_split: {args.test_split}")
    
    # Test custom splits
    args2 = parser.parse_args(['--data_dir', './test', '--val_split', '0.15', '--test_split', '0.15'])
    print(f"\n✅ Custom splits work:")
    print(f"  - val_split: {args2.val_split}")
    print(f"  - test_split: {args2.test_split}")
    print(f"  - train_split: {1 - args2.val_split - args2.test_split:.1%}")
