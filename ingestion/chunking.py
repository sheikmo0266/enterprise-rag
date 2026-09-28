from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(document["text"])

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "page": document["page"],
                "source": document["source"]
            })

    return chunks