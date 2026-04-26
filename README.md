## Final project by Idit Tikotzki and Maayan Ohayon 
The project is based on the Yode-Segmentation model (https://github.com/OneChorm/YoDe-Segmentation), released in 2023. 

Script to create a synthetic dataset: SMILES_to_IMGs_and_MASKs.py 

## Usage
### Step 1: Environ building
Copy this into terminal / Anaconda Prompt:

git clone https://github.com/OneChorm/YoDe-Segmentation.git
cd YoDe-Segmentation

conda create -n yode_seg python=3.9 -y
conda activate yode_seg

pip install torch torchvision torchaudio

pip install -r requirements.txt
pip install -e .

### Step 2: Weight configuration
Download model weights:
- Deeplabv3: Direct Download Link: https://drive.google.com/uc?id=1Sjt8McPdv7cISn-EQbU-Tbd6F5OZ45vL
- YOLOv5: Direct Download Link: https://drive.google.com/uc?id=1zxmlosjjdy8c-i7K7GmaNizGiTsBjmXW

Put them in the Weights folder.

### Step 3: Datasets
Upload the articles to test_pdf folder to convert pdf files to images and run pdf2img.py script.

You can also choose to run pdf2img.py in a python compiler.


### Step 4: Predict
run the predict_molecular.py


