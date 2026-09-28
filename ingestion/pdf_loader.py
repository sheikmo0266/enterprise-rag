import os
from pypdf import PdfReader


def load_pdfs(folder_path):

    documents = []

    for file_name in os.listdir(folder_path):

        if file_name.endswith(".pdf"):

            file_path = os.path.join(folder_path, file_name)

            reader = PdfReader(file_path)

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                documents.append({
                    "text": text,
                    "page": page_number,
                    "source": file_name
                })

    return documents