import fitz  # pymupdf
import os

if not os.path.exists("./test_img"):
    os.makedirs("./test_img")

for file in os.listdir("./test_pdf"):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join("./test_pdf", file)
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            output_path = os.path.join("./test_img", f"{os.path.splitext(file)[0]}-{page_num}.png")
            pix.save(output_path)
        doc.close()
