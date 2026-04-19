#!/bin/bash

# Generate 100,000 images and masks from SMILES data
python3 SMILES_to_IMGs_and_MASKs.py \
  --n 100000 \
  --outdir dataset \
  --max-atoms 15 \
  --smiles-file pubchem-10m.txt \
  --batch-size 500 \
  --train-ratio 0.9 \
  --split-seed 42
