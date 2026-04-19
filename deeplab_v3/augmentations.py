"""
augmentations.py

Three color augmentation types for label masks (0=background, 1=molecule):

A. Background color augmentation:
   - Recolor ONLY mask==0 pixels to a random RGB color
   - Chosen color must NOT already exist in the image
   - Default probability: 0.3

B. Partial molecule color augmentation:
   - Recolor a connected region of mask==1 pixels (not entire molecule)
   - Selected region is a contiguous connected component or partial connected component
   - Default probability: 0.6

C. Both augmentation (B then A):
   - Apply B (partial molecule), then apply A (background)
   - When applying A, new background color must not exist in the modified image
   - Default probability: 0.4

The mask is used ONLY as reference (never modified).
"""

import random
from typing import Tuple, Union, Set, Tuple as RGB
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

#from fallback_logger import get_logger

# =====================
# Utilities
# =====================

def to_rgb_array(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Convert image to uint8 RGB numpy array."""
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)

    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Unsupported image array shape: {arr.shape}")


def to_mask_array(mask: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Convert mask to uint8 2D numpy array (single channel)."""
    if isinstance(mask, Image.Image):
        return np.array(mask.convert("L"), dtype=np.uint8)

    arr = np.array(mask, dtype=np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # If RGB/RGBA, convert to grayscale-like by taking max (preserves label structure)
        arr3 = arr[..., :3]
        return np.max(arr3, axis=2).astype(np.uint8)
    raise ValueError(f"Unsupported mask array shape: {arr.shape}")


def random_rgb_color() -> RGB:
    """Generate a random RGB color."""
    return tuple(random.randint(0, 255) for _ in range(3))


def get_image_colors(img_arr: np.ndarray) -> Set[RGB]:
    """Extract all unique RGB colors currently in the image."""
    # More efficient way to get unique colors
    h, w = img_arr.shape[:2]
    # Reshape to (h*w, 3) and find unique rows
    reshaped = img_arr.reshape(-1, 3)
    unique_colors = np.unique(reshaped, axis=0)
    return set(tuple(color) for color in unique_colors)


def get_random_color_not_in_image(img_arr: np.ndarray, max_attempts: int = 100) -> RGB:
    """
    Generate a random RGB color that does NOT already exist in the image.
    If max_attempts exhausted, return a random color anyway.
    """
    existing_colors = get_image_colors(img_arr)
    for _ in range(max_attempts):
        color = random_rgb_color()
        if color not in existing_colors:
            return color
    # Fallback: return a random color
    return random_rgb_color()


def connected_partial_region(molecule_mask: np.ndarray, min_ratio: float = 0.1, max_ratio: float = 0.8, max_iters: int = 600) -> np.ndarray:
    """
    Select a contiguous region within the molecule pixels.

    Steps:
    1. Find connected components in molecule_mask
    2. Pick one component at random
    3. From that component, grow a connected region from a random seed
       until reaching a target size between min_ratio and max_ratio of the component area

    Returns: boolean mask of selected region
    """
    try:
        labeled, num_components = ndimage.label(molecule_mask)
        if num_components == 0:
            return np.zeros_like(molecule_mask, dtype=bool)

        # Pick a random component
        comp_id = random.randint(1, num_components)
        comp = (labeled == comp_id)
        area = int(comp.sum())

        if area <= 1:
            return comp

        # Sanitize ratios
        min_r = max(0.0, min(1.0, float(min_ratio)))
        max_r = max(0.0, min(1.0, float(max_ratio)))
        if max_r < min_r:
            min_r, max_r = max_r, min_r

        # Target size: random ratio of component area
        target = max(1, int(round(area * random.uniform(min_r, max_r))))

        # Grow region from random seed within the component
        ys, xs = np.where(comp)
        if len(ys) == 0:
            return comp

        seed_idx = random.randrange(len(ys))
        region = np.zeros_like(comp, dtype=bool)
        region[ys[seed_idx], xs[seed_idx]] = True

        # Expand region via binary dilation constrained to component
        for _ in range(min(max_iters, 100)):  # Limit iterations to prevent infinite loops
            if int(region.sum()) >= target:
                break
            expanded = ndimage.binary_dilation(region) & comp
            if np.array_equal(expanded, region):
                break
            region = expanded

        return region
    except Exception as e:
        # Fallback: return a simple region to avoid crashes
        #logger = get_logger()
        #mask_shape = molecule_mask.shape if hasattr(molecule_mask, 'shape') else None
        #logger.log_fallback("connected_partial_region", e, target_info=mask_shape)
        return np.zeros_like(molecule_mask, dtype=bool)


# =====================
# Augmentations A, B, C
# =====================

class AugmentationA:
    """
    A. Background color augmentation
    
    Recolor ONLY mask==0 (background) pixels to a random RGB color.
    The chosen color must NOT already exist anywhere in the image.
    """

    def __init__(self, p: float = 0.3):
        self.p = float(p)

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() > self.p:
            return img, mask

        try:
            img_arr = to_rgb_array(img)
            mask_arr = to_mask_array(mask)

            # Find background pixels (mask == 0)
            bg_mask = (mask_arr == 0)
            if not bg_mask.any():
                return Image.fromarray(img_arr), mask

            # Get a color that doesn't already exist in the image
            color = get_random_color_not_in_image(img_arr)

            # Recolor background pixels
            img_arr[bg_mask] = color

            return Image.fromarray(img_arr), mask
        except Exception as e:
            #logger = get_logger()
            #img_shape = img.size if hasattr(img, 'size') else (None, None)
            #mask_shape = mask.size if hasattr(mask, 'size') else (None, None)
            #logger.log_fallback("AugmentationA", e, 
            #                  image_info=img_shape,
            #                  target_info=mask_shape)
            return img, mask


class AugmentationB:
    """
    B. Partial molecule color augmentation
    
    Recolor a connected region of mask==1 (molecule) pixels.
    The region is a contiguous subset, not the entire molecule.
    """

    def __init__(self, p: float = 0.6, min_ratio: float = 0.1, max_ratio: float = 0.8):
        self.p = float(p)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() > self.p:
            return img, mask

        try:
            img_arr = to_rgb_array(img)
            mask_arr = to_mask_array(mask)

            # Find molecule pixels (mask == 1)
            mol_mask = (mask_arr == 1)
            if not mol_mask.any():
                return Image.fromarray(img_arr), mask

            # Select a connected partial region
            region = connected_partial_region(mol_mask, self.min_ratio, self.max_ratio)
            if not region.any():
                return Image.fromarray(img_arr), mask

            # Recolor the selected region
            color = random_rgb_color()
            img_arr[region] = color

            return Image.fromarray(img_arr), mask
        except Exception as e:
            #logger = get_logger()
            #img_shape = img.size if hasattr(img, 'size') else (None, None)
            #mask_shape = mask.size if hasattr(mask, 'size') else (None, None)
            #logger.log_fallback("AugmentationB", e,
            #                  image_info=img_shape,
            #                  target_info=mask_shape)
            return img, mask


class AugmentationC:
    """
    C. Both augmentation: apply B (partial molecule) then A (background)
    
    The background color chosen when applying A must NOT already exist
    in the image at that time (i.e., after B has been applied).
    """

    def __init__(self, p: float = 0.4, p_b: float = 0.6, p_a: float = 0.3,
                 min_ratio: float = 0.1, max_ratio: float = 0.8):
        self.p = float(p)
        self.aug_b = AugmentationB(p=p_b, min_ratio=min_ratio, max_ratio=max_ratio)
        self.aug_a = AugmentationA(p=p_a)

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() > self.p:
            return img, mask

        try:
            # Apply B first (partial molecule)
            img, mask = self.aug_b(img, mask)
            # Then apply A (background)
            img, mask = self.aug_a(img, mask)

            return img, mask
        except Exception as e:
            #logger = get_logger()
            #img_shape = img.size if hasattr(img, 'size') else (None, None)
            #mask_shape = mask.size if hasattr(mask, 'size') else (None, None)
            #logger.log_fallback("AugmentationC", e,
            #                  image_info=img_shape,
            #                  target_info=mask_shape)
            return img, mask


# =====================
# Training wrappers
# =====================

class RandomColorAugmentation:
    """
    Training wrapper: randomly selects one of the three augmentations (A, B, C).
    
    This maintains backward compatibility with training scripts that import RandomColorAugmentation.
    """

    def __init__(
        self,
        p: float = 0.5,
        background_id: int = 0,
        molecule_id: int = 1,
        mask_type: str = "labels",
        p_aug_a: float = 0.3,
        p_aug_b: float = 0.6,
        p_aug_c: float = 0.4,
    ):
        """
        Args:
            p: Overall probability to apply any augmentation
            background_id: Ignored (kept for backward compat)
            molecule_id: Ignored (kept for backward compat)
            mask_type: Ignored (kept for backward compat)
            p_aug_a: Probability for AugmentationA (background)
            p_aug_b: Probability for AugmentationB (partial molecule)
            p_aug_c: Probability for AugmentationC (both)
        """
        self.p = float(p)
        self.aug_a = AugmentationA(p=p_aug_a)
        self.aug_b = AugmentationB(p=p_aug_b)
        self.aug_c = AugmentationC(p=p_aug_c, p_a=p_aug_a, p_b=p_aug_b)

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() > self.p:
            return img, mask

        # Randomly choose one of the three augmentations
        choice = random.choice(['A', 'B', 'C'])
        if choice == 'A':
            return self.aug_a(img, mask)
        elif choice == 'B':
            return self.aug_b(img, mask)
        else:  # 'C'
            return self.aug_c(img, mask)


# Backward compatibility aliases
class RandomBackgroundColor(AugmentationA):
    """Alias for AugmentationA (background coloring)."""
    pass


class RandomMoleculeColor(AugmentationB):
    """Alias for AugmentationB (partial molecule coloring)."""
    
    def __init__(self, p: float = 0.6, min_ratio: float = 0.1, max_ratio: float = 0.8, **kwargs):
        """Accept kwargs for backward compatibility."""
        super().__init__(p=p, min_ratio=min_ratio, max_ratio=max_ratio)
