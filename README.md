## Final project by Idit Tikotzki and Maayan Ohayon 
The project is based on the Yode-Segmentation model (https://github.com/OneChorm/YoDe-Segmentation), released in 2023. 

Script to create a synthetic dataset: SMILES_to_IMGs_and_MASKs.py 

## Usage
### Step 1: Environ building
git clone https://github.com/iditberko-gif/Deep-Learning-2026-Final-Project.git

conda create -n YoDe_Seg python==3.10.11 -y
conda activate YoDe_Seg

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

cd Deep-Learning-2026-Final-Project

pip install -r requirements.txt
pip install YoDe-Segmentation-v2==1.0.1
pip install setuptools==65.5.0
pip install matplotlib==3.7.1 seaborn==0.12.2
pip install PyPDF2==3.0.1

### Step 2: Weight configuration
Download model weights:
- Deeplabv3: Direct Download Link: https://drive.google.com/uc?id=1Sjt8McPdv7cISn-EQbU-Tbd6F5OZ45vL
- YOLOv5: Direct Download Link: https://drive.google.com/uc?id=1zxmlosjjdy8c-i7K7GmaNizGiTsBjmXW

Put them in the Weights folder.

### Step 3: Data Preprocessing
Place the input PDF articles inside the test_pdf directory.
These files will be converted into images for further processing.
To perform the conversion, run:
python pdf2img.py


### Step 4: Inference and Prediction
After generating the images, run the inference pipeline:
python predict_molecular_fix.py

### NOTES
Ensure that the following directories exist before running:
test_pdf/ – contains input PDF files
test_img/ – will contain generated images
Make sure the required model weights are placed in the weights/ directory:
best.pt (YOLOv5)
model82.pth (DeepLabv3)

