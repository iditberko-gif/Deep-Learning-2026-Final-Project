# Label Mask Color Augmentations for YoDe-Segmentation

## Overview

Three color augmentation types for label masks where:
- **0 = background**
- **1 = molecule**

Augmentations **never modify the mask**, only the image pixels.

---

## Augmentation Types

### A. Background Color Augmentation

**Purpose:** Recolor ONLY background pixels (mask==0).

**Method:** 
1. Identify all pixels where mask value is 0
2. Generate a random RGB color that does **not already exist** anywhere in the image
3. Recolor all background pixels to this new color

**Probability (default):** 0.3

**Example output:** Background changes to a novel color, molecules unchanged.

---

### B. Partial Molecule Color Augmentation

**Purpose:** Recolor a connected region of molecule pixels, not the entire molecule.

**Method:**
1. Identify all molecule pixels (mask==1)
2. Find connected components in the molecule pixels
3. Randomly pick one component
4. From that component, grow a connected region from a random seed pixel
5. Grow until reaching a random target size (10–80% of the component area)
6. Recolor only that selected region with a random RGB color

**Probability (default):** 0.6

**Example output:** Part of a molecule changes color; other molecule parts and background unchanged.

---

### C. "Both" Augmentation (B then A)

**Purpose:** Apply B (partial molecule) then A (background).

**Method:**
1. Apply augmentation B to the image
2. Then apply augmentation A **on the result** (after B has already modified it)
3. When applying A, the new background color is chosen to not exist in the **already-augmented image**

**Probability (default):** 0.4

**Ordering:** B before A (molecule first, then background).

**Example output:** Part of molecule changes color, then background changes to a novel color (not present after B was applied).

---

## Implementation

### File: `deeplab_v3/augmentations.py`

**Core classes:**
- `AugmentationA(p=0.3)` — Background augmentation
- `AugmentationB(p=0.6, min_ratio=0.1, max_ratio=0.8)` — Partial molecule augmentation
- `AugmentationC(p=0.4, p_a=0.3, p_b=0.6, min_ratio=0.1, max_ratio=0.8)` — Both augmentation

**Training wrapper:**
- `RandomColorAugmentation(p=0.5, p_aug_a=0.3, p_aug_b=0.6, p_aug_c=0.4)` 
  - Randomly selects one of A, B, or C with equal probability
  - Used in training via the `SegmentationPresetTrain` class

**Helper functions:**
- `to_rgb_array(img)` — Convert image to uint8 RGB numpy array
- `to_mask_array(mask)` — Convert mask to uint8 2D numpy array
- `get_random_color_not_in_image(img_arr, max_attempts=100)` — Sample a color not in the image
- `connected_partial_region(molecule_mask, min_ratio, max_ratio)` — Select partial connected region

### Usage in Training

Applied in the transform pipeline **before** `ToTensor()/Normalize()`:

```
Resize → HorizontalFlip → RandomColorAugmentation → ToTensor() → Normalize()
```

In `train_multi_GPU.py`:
```python
RandomColorAugmentation(
    p=color_aug_prob,              # Overall probability (0.5)
    p_aug_a=0.3,                   # Probability for augmentation A
    p_aug_b=color_aug_prob,        # Probability for augmentation B
    p_aug_c=0.4,                   # Probability for augmentation C
)
```

---

## Testing

Run the sanity test to verify all three augmentations work correctly:

```bash
cd deeplab_v3
python test_augmentations_sanity.py \
    --images-dir ./dataset/VOCdevkit/VOC2012/JPEGImages \
    --masks-dir ./dataset/VOCdevkit/VOC2012/SegmentationClass \
    --num-samples 3 \
    --save-dir ./tmp_sanity_test
```

The test:
- Loads sample images and masks
- Applies A, B, and C with p=1.0 (always) to each sample
- Verifies that:
  - **A** only modifies mask==0 pixels
  - **B** only modifies mask==1 pixels  
  - **C** only modifies mask==0 and mask==1 pixels
- Saves originals and all three augmented versions for visual inspection

---

## Customization

Edit probabilities in `train_multi_GPU.py`:

```python
RandomColorAugmentation(
    p=0.8,              # Overall probability (increase for more augmentation)
    p_aug_a=0.2,        # Background augmentation frequency
    p_aug_b=0.5,        # Partial molecule augmentation frequency
    p_aug_c=0.3,        # Both augmentation frequency
)
```

Or use individual augmentations:

```python
from augmentations import AugmentationA, AugmentationB, AugmentationC

aug_a = AugmentationA(p=0.4)
aug_b = AugmentationB(p=0.6, min_ratio=0.05, max_ratio=0.9)
aug_c = AugmentationC(p=0.5, p_a=0.4, p_b=0.6)
```

---

## Mask Format

The mask is a single-channel label mask (uint8):
- **Pixel value 0:** background
- **Pixel value 1:** molecule (or target foreground)

The augmentations read from the mask to decide which image pixels to recolor; the mask itself is never modified.

---

## Default Probabilities

Based on the requirement specification:
- **p_aug_a (Background):** 0.3
- **p_aug_b (Partial Molecule):** 0.6
- **p_aug_c (Both):** 0.4

These can be tuned based on empirical training results.
