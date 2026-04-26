import fitz  # PyMuPDF
import os

# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "test_pdf")
IMG_DIR = os.path.join(BASE_DIR, "test_img")

# Create output folder if it doesn't exist
os.makedirs(IMG_DIR, exist_ok=True)

# Check if PDF folder exists
if not os.path.exists(PDF_DIR):
    raise FileNotFoundError(f"Folder not found: {PDF_DIR}")

# Process PDFs
for file in os.listdir(PDF_DIR):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, file)

        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()

            output_path = os.path.join(
                IMG_DIR,
                f"{os.path.splitext(file)[0]}-{page_num}.png"
            )

            pix.save(output_path)

        doc.close()

print("Done converting PDFs to images.")
