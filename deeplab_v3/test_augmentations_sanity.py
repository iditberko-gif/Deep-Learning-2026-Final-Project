#!/usr/bin/env python3
"""
test_augmentations_sanity.py

Sanity test demonstrating all three augmentation types (A, B, C) on the sample dataset.

Requirements:
- Augmentation A: recolor mask==0 (background) pixels only, color must not exist in image
- Augmentation B: recolor a connected partial region of mask==1 (molecule) pixels
- Augmentation C: apply B then A (both)

Output: saves original, A-augmented, B-augmented, C-augmented versions for visual inspection
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from augmentations import AugmentationA, AugmentationB, AugmentationC


def get_sample_images(images_dir, masks_dir, num_samples=5):
    """Load image-mask pairs from the dataset."""
    pairs = []
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])[:num_samples]
    for img_file in img_files:
        base = img_file[:-4]  # remove .jpg
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, base + '.png')
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path, base))
    return pairs


def verify_augmentation_a(original_arr, augmented_arr, mask_arr):
    """Verify that A only modified background pixels (mask==0)."""
    bg_mask = (mask_arr == 0)
    # Check that only background pixels changed
    changed = (original_arr != augmented_arr).any(axis=2)
    changed_and_background = (changed & bg_mask).sum()
    changed_not_background = (changed & ~bg_mask).sum()
    
    print(f"  A: Background pixels changed: {changed_and_background}, "
          f"Non-background pixels changed: {changed_not_background}")
    if changed_not_background > 0:
        print("  WARNING: A modified non-background pixels!")
    return changed_not_background == 0


def verify_augmentation_b(original_arr, augmented_arr, mask_arr):
    """Verify that B only modified molecule pixels (mask==1)."""
    mol_mask = (mask_arr == 1)
    # Check that only molecule pixels changed
    changed = (original_arr != augmented_arr).any(axis=2)
    changed_and_molecule = (changed & mol_mask).sum()
    changed_not_molecule = (changed & ~mol_mask).sum()
    
    print(f"  B: Molecule pixels changed: {changed_and_molecule}, "
          f"Non-molecule pixels changed: {changed_not_molecule}")
    if changed_not_molecule > 0:
        print("  WARNING: B modified non-molecule pixels!")
    return changed_not_molecule == 0


def verify_augmentation_c(original_arr, augmented_arr, mask_arr):
    """Verify that C only modified background and molecule pixels."""
    mol_mask = (mask_arr == 1)
    bg_mask = (mask_arr == 0)
    # Check that only background or molecule pixels changed
    changed = (original_arr != augmented_arr).any(axis=2)
    changed_and_valid = (changed & (mol_mask | bg_mask)).sum()
    changed_invalid = (changed & ~(mol_mask | bg_mask)).sum()
    
    print(f"  C: Valid pixels changed (bg or mol): {changed_and_valid}, "
          f"Invalid pixels changed: {changed_invalid}")
    if changed_invalid > 0:
        print("  WARNING: C modified pixels outside bg/mol!")
    return changed_invalid == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="./dataset/VOCdevkit/VOC2012/JPEGImages")
    ap.add_argument("--masks-dir", default="./dataset/VOCdevkit/VOC2012/SegmentationClass")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--save-dir", default="./tmp_sanity_test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load sample pairs
    pairs = get_sample_images(args.images_dir, args.masks_dir, args.num_samples)
    if not pairs:
        print("ERROR: No image-mask pairs found!")
        sys.exit(1)

    print(f"Found {len(pairs)} pairs")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize augmentations
    aug_a = AugmentationA(p=1.0)  # Force p=1 for testing
    aug_b = AugmentationB(p=1.0, min_ratio=0.1, max_ratio=0.8)
    aug_c = AugmentationC(p=1.0, p_a=1.0, p_b=1.0)

    all_pass = True

    for img_path, mask_path, base in pairs:
        print(f"\nProcessing {base}...")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert to arrays for verification
        img_arr = np.array(img, dtype=np.uint8)
        mask_arr = np.array(mask.convert("L"), dtype=np.uint8)

        # Apply augmentation A
        print("  Applying A (background)...")
        aug_a_img, _ = aug_a(img, mask)
        aug_a_arr = np.array(aug_a_img, dtype=np.uint8)
        pass_a = verify_augmentation_a(img_arr, aug_a_arr, mask_arr)
        all_pass = all_pass and pass_a

        # Apply augmentation B
        print("  Applying B (partial molecule)...")
        aug_b_img, _ = aug_b(img, mask)
        aug_b_arr = np.array(aug_b_img, dtype=np.uint8)
        pass_b = verify_augmentation_b(img_arr, aug_b_arr, mask_arr)
        all_pass = all_pass and pass_b

        # Apply augmentation C (B then A)
        print("  Applying C (both: B then A)...")
        aug_c_img, _ = aug_c(img, mask)
        aug_c_arr = np.array(aug_c_img, dtype=np.uint8)
        pass_c = verify_augmentation_c(img_arr, aug_c_arr, mask_arr)
        all_pass = all_pass and pass_c

        # Save outputs
        img.save(os.path.join(args.save_dir, f"{base}_00_original.png"))
        aug_a_img.save(os.path.join(args.save_dir, f"{base}_01_A_background.png"))
        aug_b_img.save(os.path.join(args.save_dir, f"{base}_02_B_partial_molecule.png"))
        aug_c_img.save(os.path.join(args.save_dir, f"{base}_03_C_both.png"))

        print(f"  Saved to {args.save_dir}/")

    print("\n" + "=" * 60)
    if all_pass:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED - check warnings above")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
