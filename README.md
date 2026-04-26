## Final project by Idit Tikotzki and Maayan Ohayon 
The project is based on the Yode-Segmentation model (https://github.com/OneChorm/YoDe-Segmentation), released in 2023. 

Script to create a synthetic dataset: SMILES_to_IMGs_and_MASKs.py 

## Usage
### Step 1: Environ building
```bash
git clone https://github.com/OneChorm/YoDe-Segmentation
conda create -n YoDe_Seg python==3.10.11
conda activate YoDe_Seg
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd "your dir to `YoDe-Segmentation`" # change directory
pip install -r requirements.txt
pip install YoDe-Segmentation-v2==1.0.1
```

### Step 2: Weight configuration
Download model weights:
- Deeplabv3: https://drive.google.com/file/d/1Sjt8McPdv7cISn-EQbU-Tbd6F5OZ45vL/view?usp=drive_link
- YOLOv5: https://drive.google.com/file/d/1zxmlosjjdy8c-i7K7GmaNizGiTsBjmXW/view?usp=drive_link 
Put them in the Weights folder.

### Step 3: 

### Step 4: Predict
run the predict_molecular.py
```bash
python predict_molecular.py
```


