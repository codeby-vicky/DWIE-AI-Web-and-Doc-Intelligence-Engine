import fitz
import re
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document

from core.image_analyzer import analyze_image_with_ollama
from core.table_extractor import extract_tables_from_pdf


# -------------------------
# TEXT EXTRACTION
# -------------------------
def extract_page_text(page):
    text = page.get_text("text")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------
# IMAGE EXTRACTION
# -------------------------
def extract_images_from_pdf(doc, source_name):

    image_documents = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img in image_list:
            xref = img[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", None)

                if not image_bytes:
                    continue

                description = analyze_image_with_ollama(image_bytes)

                image_documents.append(
                    Document(
                        page_content=f"Image found on page {page_index + 1}:\n{description}",
                        metadata={
                            "source": source_name,
                            "type": "image",
                            "page": page_index + 1,
                        },
                    )
                )

            except Exception:
                continue

    return image_documents


# -------------------------
# MAIN MULTIMODAL EXTRACTION
# -------------------------
def extract_all_content(pdf_docs):

    all_documents = []

    for pdf in pdf_docs:

        pdf.seek(0)
        pdf_bytes = pdf.read()

        # ---- TEXT ----
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        with ThreadPoolExecutor() as executor:
            page_texts = list(executor.map(extract_page_text, doc))

        for page_number, page_text in enumerate(page_texts):
            if page_text:
                all_documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": pdf.name,
                            "type": "text",
                            "page": page_number + 1,
                        },
                    )
                )

        # ---- IMAGES ----
        image_docs = extract_images_from_pdf(doc, pdf.name)
        all_documents.extend(image_docs)

        doc.close()

        # ---- TABLES ----
        pdf.seek(0)
        table_docs = extract_tables_from_pdf(pdf)
        all_documents.extend(table_docs)

    return all_documents
