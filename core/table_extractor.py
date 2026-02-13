import pdfplumber
from langchain_core.documents import Document


def extract_tables_from_pdf(pdf_file):
    """
    Extract tables using pdfplumber
    Converts tables into structured text
    """

    table_documents = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_index, page in enumerate(pdf.pages):
                tables = page.extract_tables()

                for table in tables:
                    if not table:
                        continue

                    table_text = "\n".join(
                        [
                            " | ".join([cell if cell else "" for cell in row])
                            for row in table
                        ]
                    )

                    table_documents.append(
                        Document(
                            page_content=f"Table found on page {page_index + 1}:\n{table_text}",
                            metadata={
                                "source": getattr(pdf_file, "name", "uploaded_pdf"),
                                "type": "table",
                                "page": page_index + 1,
                            },
                        )
                    )

    except Exception:
        pass

    return table_documents
