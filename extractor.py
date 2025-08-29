import io
import pdfplumber
import easyocr
from fastapi import UploadFile

reader = easyocr.Reader(['en'], gpu=False)  

def extract_text(file: UploadFile):
    content = file.file.read()
    filename = file.filename.lower()

    # Text file
    if filename.endswith(".txt"):
        return content.decode("utf-8")

    # PDF file
    elif filename.endswith(".pdf"):
        # Try normal PDF text first
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [page.extract_text() for page in pdf.pages]
                text = "\n".join(pages)
            if text.strip():
                return text
        except Exception:
            pass
        # If empty or fails → OCR
        return ocr_pdf(content)

    # Images
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return ocr_image(content)

    else:
        raise ValueError("Unsupported file type")

def ocr_image(image_bytes):
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = reader.readtext(np.array(img))
    return " ".join([res[1] for res in results])

def ocr_pdf(pdf_bytes):
    from pdf2image import convert_from_bytes
    import numpy as np
    from PIL import Image

    images = convert_from_bytes(pdf_bytes)
    all_text = []
    for img in images:
        img = img.convert("RGB")
        results = reader.readtext(np.array(img))
        all_text.extend([res[1] for res in results])
    return " ".join(all_text)
