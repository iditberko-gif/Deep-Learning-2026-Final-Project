## Final project by Idit Tikotzki and Maayan Ohayon 
The project is based on the Yode-Segmentation model (https://github.com/OneChorm/YoDe-Segmentation), released in 2023. 

Script to create a synthetic dataset: SMILES_to_IMGs_and_MASKs.py 

## Usage
### Step 1: Environ building
Copy this into terminal / Anaconda Prompt:
1. Clone repo
git clone https://github.com/OneChorm/YoDe-Segmentation.git
cd YoDe-Segmentation

2. Create clean environment
conda create -n yode_seg python=3.9 -y
conda activate yode_seg

3. Install PyTorch (CUDA 11.8 stable version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. Install project dependencies
pip install -r requirements.txt

5. Install repo as editable package (safe replacement)
pip install -e .

### Step 2: Weight configuration
Download model weights:
- Deeplabv3: Direct Download Link: https://drive.google.com/uc?id=1Sjt8McPdv7cISn-EQbU-Tbd6F5OZ45vL
- YOLOv5: Direct Download Link: https://drive.google.com/uc?id=1zxmlosjjdy8c-i7K7GmaNizGiTsBjmXW

Put them in the Weights folder.

### Step 3: Datasets
If you need to work with pdf files, you should put the files in the test_pdf folder,and then convert pdf files to images. If you run into an error with the pdf conversion, you could run the pdf2img.py in a python compiler.

run the pdf2img.py

python pdf2img.py
if need to work with images, you should only put the images in the test_img folder

### Step 4: Predict
run the predict_molecular.py
```bash
python predict_molecular.py
```


