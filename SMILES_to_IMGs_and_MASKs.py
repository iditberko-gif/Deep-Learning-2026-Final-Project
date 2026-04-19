#!/usr/bin/env python3
"""
Read SMILES from file, canonicalize, filter small molecules (<= 15 atoms),
render molecule images with RDKit, and generate binary masks.

For each molecule:
- image: RGB PNG
- mask: binary PNG (1 = molecule, 0 = background)
"""

# argparse: For parsing command-line arguments to configure the script's behavior, such as input/output paths and parameters.
import argparse
# os: Provides functions for interacting with the operating system, used for file and directory operations like creating directories and checking file existence.
import os
# typing: Provides type hints for better code documentation and static type checking, used for annotations like Optional, List, Tuple.
from typing import Optional, List, Tuple
# dataclasses: Allows creation of simple classes for storing data, used here for the MoleculeRecord dataclass to hold molecule information.
from dataclasses import dataclass
# numpy: A library for numerical computing, used for array operations, particularly for image processing and mask generation.
import numpy as np
# PIL (Pillow): Python Imaging Library, used for image manipulation, such as converting and saving images in PNG format.
from PIL import Image
import random
from typing import Sequence
# tqdm: Progress bar library for displaying progress during long-running operations.
from tqdm import tqdm

# rdkit: Open-source cheminformatics library, core module for handling chemical molecules.
from rdkit import Chem
# rdkit.Chem.Draw: Submodule of RDKit for drawing molecular structures, used to render 2D images of molecules.
from rdkit.Chem import Draw
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDepictor


# =========================
# Canonicalize SMILES
# =========================
def canonicalize(smiles: str) -> str:
    """
    Canonicalize SMILES string using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # Return original if invalid
    return Chem.MolToSmiles(mol, True)


# =========================
# Simple container for molecule metadata
# =========================
@dataclass
class MoleculeRecord:
    cid: int
    smiles: str
    num_atoms: int


# =========================
# Fetch Canonical SMILES for multiple CIDs in batch
# =========================
# =========================
# Convert SMILES to RDKit Mol + atom count
# =========================
def smiles_to_mol(smiles: str) -> Tuple[Optional[Chem.Mol], int]:
    """
    Parse SMILES using RDKit.
    Returns molecule and number of atoms (graph atoms).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 0
    return mol, mol.GetNumAtoms()


# =========================
# Render molecule image and binary mask
# =========================
def render_image_and_mask(
    mol: Chem.Mol,
    img_size: int = 300
) -> Tuple[Image.Image, Image.Image]:
    """
    Render molecule using RDKit and generate a binary mask.

    Mask logic:
    - Convert RGB image to numpy array
    - Background is white (255,255,255)
    - Any pixel != white → molecule (mask = 1)
    """
    # Orient molecule and render image (PIL Image)
    mol_to_draw = orient_molecule_flat(mol)
    img = Draw.MolToImage(mol_to_draw, size=(img_size, img_size))

    # Convert to numpy array (H, W, 3)
    arr = np.array(img)

    # Detect non-white pixels; binary mask values 0 (background) or 1 (molecule)
    mask = np.any(arr < 250, axis=2).astype(np.uint8)

    # Convert mask back to PIL (single-channel)
    mask_img = Image.fromarray(mask, mode="L")

    return img, mask_img


def _render_fragment_rgba(fragment_mol: Chem.Mol, img_size: int = 300) -> Tuple[Image.Image, Image.Image]:
    """
    Render a fragment as an RGBA image and its alpha mask (L mode) where non-white → opaque.
    Returns (rgba_img, alpha_mask).
    """
    # Orient fragment then render on white background
    frag_to_draw = orient_molecule_flat(fragment_mol)
    frag_img = Draw.MolToImage(frag_to_draw, size=(img_size, img_size))
    frag_arr = np.array(frag_img)
    alpha = np.any(frag_arr < 250, axis=2).astype(np.uint8) * 255
    alpha_img = Image.fromarray(alpha, mode="L")
    rgba = frag_img.convert("RGBA")
    rgba.putalpha(alpha_img)
    return rgba, alpha_img


def extract_random_fragment(source: Chem.Mol, max_fragment_atoms: Optional[int] = None) -> Optional[Chem.Mol]:
    """
    Break random bonds in `source` and return a random fragment Mol.
    Returns None when no suitable fragment is found.
    """
    nb = source.GetNumBonds()
    if nb == 0:
        return None

    # try breaking between 1 and min(3, nb) bonds
    k = random.randint(1, min(3, nb))
    bond_indices = random.sample(list(range(nb)), k)

    try:
        frag_on_bonds = Chem.FragmentOnBonds(source, bond_indices, addDummies=False)
    except Exception:
        return None

    frags = Chem.GetMolFrags(frag_on_bonds, asMols=True)
    if not frags:
        return None

    # Filter fragments: exclude trivial or full-molecule fragments
    cand = [f for f in frags if 0 < f.GetNumAtoms() < source.GetNumAtoms()]
    if max_fragment_atoms is not None:
        cand = [f for f in cand if f.GetNumAtoms() <= max_fragment_atoms]
    if not cand:
        return None

    return random.choice(cand)


def orient_molecule_flat(mol: Chem.Mol) -> Chem.Mol:
    """
    Return a copy of `mol` with 2D coordinates computed and rotated
    so the principal axis is aligned horizontally (x-axis). This produces
    consistent, article-like horizontal depictions.
    """
    if mol is None:
        return mol
    m = Chem.Mol(mol)
    try:
        rdDepictor.Compute2DCoords(m)
    except Exception:
        pass

    conf = m.GetConformer()
    n = m.GetNumAtoms()
    if n == 0:
        return m

    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(n)], dtype=float)
    # center
    centroid = coords.mean(axis=0)
    coords -= centroid

    # PCA via SVD to get principal axis
    try:
        u, s, vh = np.linalg.svd(coords, full_matrices=False)
        principal = vh[0]
    except Exception:
        principal = np.array([1.0, 0.0])

    angle = np.arctan2(principal[1], principal[0])
    # rotate so principal aligns with x-axis
    c = np.cos(-angle)
    s_ = np.sin(-angle)
    R = np.array([[c, -s_], [s_, c]])
    new_coords = coords.dot(R.T)

    # write back
    for i in range(n):
        x, y = new_coords[i] + centroid
        conf.SetAtomPosition(i, Point3D(x, y, 0.0))

    return m


def add_fragment_noise_to_image(base_img: Image.Image, fragment_mol: Chem.Mol, img_size: int = 300) -> Image.Image:
    """
    Overlay a rendered `fragment_mol` onto `base_img` at a random position/scale/rotation.
    Returns a new PIL Image (RGB) with the fragment composited.
    The fragment's opaque pixels replace the base image pixels (no mask modification).
    """
    # Render fragment at full size, then scale/rotate
    frag_rgba, frag_alpha = _render_fragment_rgba(fragment_mol, img_size=img_size)

    # Random scale between 0.2 and 0.7 of image
    scale = random.uniform(0.2, 0.7)
    new_w = max(1, int(img_size * scale))
    new_h = max(1, int(img_size * scale))
    frag_resized = frag_rgba.resize((new_w, new_h), resample=Image.BILINEAR)

    # No rotation requested
    frag_rot = frag_resized

    # Position: random such that fragment partly on image
    bw, bh = base_img.size
    fw, fh = frag_rot.size
    # allow negative offsets so fragment can be partially outside
    x = random.randint(-fw // 2, bw - fw // 2)
    y = random.randint(-fh // 2, bh - fh // 2)

    out = base_img.copy().convert("RGBA")
    out.alpha_composite(Image.new("RGBA", out.size, (255, 255, 255, 0)))
    # Paste using fragment alpha as mask
    out.paste(frag_rot, (x, y), frag_rot)
    return out.convert("RGB")


# =========================
# Main collection loop (batch-based)
# =========================
def collect_molecules(
    batch_size: int,
    max_atoms: int,
    smiles_file: str,
    start_line: int,
    max_attempts: Optional[int],
    seen_smiles: set
) -> Tuple[List[MoleculeRecord], int]:
    """
    Read SMILES from file starting at start_line, canonicalize, and collect valid molecules.
    Filters molecules by maximum atom count.
    Returns (records, next_line_to_start_from) to support batching.
    """
    records: List[MoleculeRecord] = []
    attempts = 0
    next_line = start_line

    with open(smiles_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num < start_line:
                continue
            if len(records) >= batch_size:
                break
            if max_attempts is not None and attempts >= max_attempts:
                break
            attempts += 1

            smiles = line.strip()
            if not smiles:
                next_line = line_num + 1
                continue

            # Canonicalize SMILES
            smiles = canonicalize(smiles)
            if smiles in seen_smiles:
                next_line = line_num + 1
                continue

            # Convert to RDKit molecule
            mol, atom_count = smiles_to_mol(smiles)
            if mol is None or atom_count > max_atoms:
                next_line = line_num + 1
                continue

            # Keep molecule
            seen_smiles.add(smiles)
            records.append(MoleculeRecord(line_num, smiles, atom_count))
            next_line = line_num + 1

    return records, next_line


# =========================
# Entry point
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="Number of molecules to fetch")
    parser.add_argument("--outdir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--max-atoms", type=int, default=15, help="Maximum atom count")
    parser.add_argument("--smiles-file", type=str, default=r"/gpfs0/bgu-anatm/users/maayanoh/YODE/pubchem-10m.txt", help="Path to SMILES file")
    parser.add_argument("--max-attempts", type=int, default=None, help="Max SMILES to process (None = no cap)")
    parser.add_argument("--add-noise", action="store_true", help="Overlay border-touching whole-molecule noise onto images")
    parser.add_argument("--noise-seed", type=int, default=None, help="Random seed for reproducible placement (None = non-deterministic)")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of molecules per batch")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Training set ratio (0.0 to 1.0)")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed for train/val split")

    args = parser.parse_args()

    # Create Pascal VOC 2012 output folders
    base_outdir = os.path.join(args.outdir, "VOCdevkit", "VOC2012")
    img_dir = os.path.join(base_outdir, "JPEGImages")
    mask_dir = os.path.join(base_outdir, "SegmentationClass")
    segmentation_set_dir = os.path.join(base_outdir, "ImageSets", "Segmentation")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(segmentation_set_dir, exist_ok=True)
    
    # Create mask visualizations folder
    mask_vis_dir = os.path.join(args.outdir, "mask_visualizations")
    os.makedirs(mask_vis_dir, exist_ok=True)

    print(f"Starting to collect {args.n} molecules (max atoms: {args.max_atoms}, from file: {args.smiles_file})")
    
    # Overall progress bar for total molecules
    overall_pbar = tqdm(total=args.n, desc="Total molecules processed", leave=True)
    
    # Track molecules collected and seen SMILES
    total_collected = 0
    seen_smiles = set()
    current_line = 1
    sample_counter = 0  # Counter for sequential image naming
    basenames = []  # Track successfully created samples
    
    # Process molecules in batches
    while total_collected < args.n:
        # Determine how many more molecules we need
        remaining = args.n - total_collected
        current_batch_size = min(args.batch_size, remaining)
        
        # Collect a batch
        batch_molecules, current_line = collect_molecules(
            batch_size=current_batch_size,
            max_atoms=args.max_atoms,
            smiles_file=args.smiles_file,
            start_line=current_line,
            max_attempts=args.max_attempts,
            seen_smiles=seen_smiles,
        )

        #if total_collected  < 35329:
        #   total_collected += len(batch_molecules)
        #  overall_pbar.update(len(batch_molecules))
        # continue
        
        if not batch_molecules:
            print(f"\nWarning: Reached end of file with only {total_collected} molecules collected (requested {args.n})")
            break
        
        # Process this batch with batch progress bar
        with tqdm(total=len(batch_molecules), desc="Batch processing", leave=False) as batch_pbar:
            # Render and save images + masks for this batch
            for rec in batch_molecules:
                # seed for reproducibility per-image (only if a seed was provided)
                if args.noise_seed is not None:
                    random.seed(args.noise_seed + rec.cid)

                # Random image size between 100 and 400 (inclusive)
                img_size = random.randint(100, 400)

                # Determine how many whole molecules to place (1..3, mostly 1)
                if random.random() < 0.3:
                    num_whole = random.randint(2, 3)
                else:
                    num_whole = 1

                # select molecules for this image (include rec as primary)
                chosen = [rec]
                candidates = [m for m in batch_molecules if m.cid != rec.cid]
                random.shuffle(candidates)
                for c in candidates:
                    if len(chosen) >= num_whole:
                        break
                    chosen.append(c)

                # Create blank RGBA canvas
                canvas = Image.new("RGBA", (img_size, img_size), (255, 255, 255, 255))

                # Track occupied mask (numpy array) to avoid overlaps with noise and other molecules
                occupied = np.zeros((img_size, img_size), dtype=np.uint8)

                # Build a single binary mask per image: 0 = background, 255 = molecule
                mask_arr = np.zeros((img_size, img_size), dtype=np.uint8)
                placed_cids = set()

                # Place each whole molecule (these should have masks)
                for i, mrec in enumerate(chosen):
                    m_mol, _ = smiles_to_mol(mrec.smiles)
                    if m_mol is None:
                        continue

                    # Render base RGBA for molecule at full canvas size
                    base_rgba, base_alpha = _render_fragment_rgba(m_mol, img_size=img_size)

                    # For the primary molecule (first in chosen), center it.
                    placed = False
                    attempts = 0
                    while not placed and attempts < 50:
                        attempts += 1
                        if i == 0:
                            # scale to fit nicely in center
                            scale = random.uniform(0.4, 0.9)
                            new_w = max(1, int(img_size * scale))
                            new_h = max(1, int(img_size * scale))
                            rg = base_rgba.resize((new_w, new_h), resample=Image.BILINEAR)
                            # no rotation per user request
                            rg = rg
                            fw, fh = rg.size
                            x = (img_size - fw) // 2
                            y = (img_size - fh) // 2
                        else:
                            # place extras randomly inside canvas (fully inside)
                            scale = random.uniform(0.3, 0.8)
                            new_w = max(1, int(img_size * scale))
                            new_h = max(1, int(img_size * scale))
                            rg = base_rgba.resize((new_w, new_h), resample=Image.BILINEAR)
                            # no rotation for extras either
                            rg = rg
                            fw, fh = rg.size
                            if fw >= img_size or fh >= img_size:
                                x = random.randint(0, max(0, img_size - 1))
                                y = random.randint(0, max(0, img_size - 1))
                            else:
                                x = random.randint(0, img_size - fw)
                                y = random.randint(0, img_size - fh)

                        # get alpha array of transformed molecule
                        alpha_arr = np.array(rg.split()[-1])
                        # check overlap with existing occupied
                        ox1 = max(0, x)
                        oy1 = max(0, y)
                        ox2 = min(img_size, x + rg.size[0])
                        oy2 = min(img_size, y + rg.size[1])
                        if ox1 >= ox2 or oy1 >= oy2:
                            continue
                        sub_alpha = alpha_arr[oy1 - y:oy2 - y, ox1 - x:ox2 - x]
                        occ_sub = occupied[oy1:oy2, ox1:ox2]
                        if np.any((sub_alpha > 0) & (occ_sub > 0)):
                            continue

                        # place it
                        canvas.paste(rg, (x, y), rg)
                        # update occupied
                        occupied[oy1:oy2, ox1:ox2] = np.maximum(occ_sub, (sub_alpha > 0).astype(np.uint8))

                        # update binary mask: set pixels of this molecule to 1
                        sel = (sub_alpha > 0)
                        mask_arr[oy1:oy2, ox1:ox2][sel] = 1
                        placed_cids.add(mrec.cid)

                        placed = True

                # Add partial molecules: whole molecules that are partially off-canvas and touch border
                # Partial molecules are NOT added to the mask
                noise_count = random.randint(0, 5)
                noise_tries = 0
                placed_noise = 0
                while placed_noise < noise_count and noise_tries < noise_count * 30:
                    noise_tries += 1
                    donor = random.choice(batch_molecules)
                    if donor.cid in placed_cids:
                        continue
                    donor_mol, _ = smiles_to_mol(donor.smiles)
                    if donor_mol is None:
                        continue

                    # render donor at base size
                    donor_rgba, donor_alpha = _render_fragment_rgba(donor_mol, img_size=img_size)

                    # choose larger scale so it can extend off-canvas
                    scale = random.uniform(0.6, 1.6)
                    new_w = max(1, int(img_size * scale))
                    new_h = max(1, int(img_size * scale))
                    dr = donor_rgba.resize((new_w, new_h), resample=Image.BILINEAR)
                    # no rotation for partial molecules
                    dr = dr
                    fw, fh = dr.size

                    # pick a side to place so part is off-canvas and touches border
                    side = random.choice(["left", "right", "top", "bottom"])
                    if side == "left":
                        x = -random.randint(1, fw // 2)
                        y = random.randint(-fh // 2, img_size - fh // 2)
                    elif side == "right":
                        x = img_size - random.randint(fw // 2, fw)
                        y = random.randint(-fh // 2, img_size - fh // 2)
                    elif side == "top":
                        y = -random.randint(1, fh // 2)
                        x = random.randint(-fw // 2, img_size - fw // 2)
                    else:
                        y = img_size - random.randint(fh // 2, fh)
                        x = random.randint(-fw // 2, img_size - fw // 2)

                    # compute overlap region
                    ox1 = max(0, x)
                    oy1 = max(0, y)
                    ox2 = min(img_size, x + fw)
                    oy2 = min(img_size, y + fh)
                    if ox1 >= ox2 or oy1 >= oy2:
                        continue

                    dr_alpha = np.array(dr.split()[-1])
                    sub_alpha = dr_alpha[oy1 - y:oy2 - y, ox1 - x:ox2 - x]

                    # must touch border: check that any alpha on image border is non-zero
                    # build a temporary full-size alpha canvas for this donor placement
                    temp_alpha = np.zeros((img_size, img_size), dtype=np.uint8)
                    temp_alpha[oy1:oy2, ox1:ox2] = (sub_alpha > 0).astype(np.uint8)
                    border_mask = np.zeros_like(temp_alpha)
                    border_mask[0, :] = 1
                    border_mask[-1, :] = 1
                    border_mask[:, 0] = 1
                    border_mask[:, -1] = 1
                    if not np.any(temp_alpha & border_mask):
                        continue

                    # ensure no overlap with occupied (molecules)
                    if np.any((temp_alpha > 0) & (occupied > 0)):
                        continue

                    # paste donor and mark occupied so other noise won't touch
                    canvas.paste(dr, (x, y), dr)
                    # update occupied with the donor's alpha region
                    occupied = np.maximum(occupied, temp_alpha)
                    # NOTE: We do NOT update mask_arr for partial molecules
                    placed_noise += 1

                # finalize image and save
                final_img = canvas.convert("RGB")

                # Generate sequential zero-padded filename
                sample_counter += 1
                basename = f"{sample_counter:06d}"
                
                # Save JPEG image
                img_path = os.path.join(img_dir, f"{basename}.jpg")
                final_img.save(img_path, format="JPEG")

                # Save mask as grayscale PNG with 0=background, 1=molecule
                mask_img = Image.fromarray(mask_arr, mode="L")
                mask_path = os.path.join(mask_dir, f"{basename}.png")
                mask_img.save(mask_path)
                
                # Save visualization copy of mask (1 -> 255, 0 -> 0)
                mask_vis_arr = mask_arr * 255
                mask_vis_img = Image.fromarray(mask_vis_arr, mode="L")
                mask_vis_path = os.path.join(mask_vis_dir, f"{basename}_vis.png")
                mask_vis_img.save(mask_vis_path)
                
                # Track successfully created sample
                basenames.append(basename)
                
                batch_pbar.update(1)
        
        # Update overall progress and totals
        total_collected += len(batch_molecules)
        overall_pbar.update(len(batch_molecules))
    
    overall_pbar.close()
    
    # Create train/val split
    random.seed(args.split_seed)
    random.shuffle(basenames)
    
    num_train = int(len(basenames) * args.train_ratio)
    train_basenames = basenames[:num_train]
    val_basenames = basenames[num_train:]
    
    # Write train.txt
    train_txt_path = os.path.join(segmentation_set_dir, "train.txt")
    with open(train_txt_path, 'w') as f:
        for basename in train_basenames:
            f.write(f"{basename}\n")
    
    # Write val.txt
    val_txt_path = os.path.join(segmentation_set_dir, "val.txt")
    with open(val_txt_path, 'w') as f:
        for basename in val_basenames:
            f.write(f"{basename}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset generation completed!")
    print(f"{'='*60}")
    print(f"Total samples generated:        {len(basenames)}")
    print(f"Training samples:               {len(train_basenames)}")
    print(f"Validation samples:             {len(val_basenames)}")
    print(f"{'='*60}")
    print(f"JPEGImages directory:           {os.path.abspath(img_dir)}")
    print(f"SegmentationClass directory:    {os.path.abspath(mask_dir)}")
    print(f"train.txt file:                 {os.path.abspath(train_txt_path)}")
    print(f"val.txt file:                   {os.path.abspath(val_txt_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
