from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "page": chunk["page"],
            "source": chunk["source"]
        }
        for chunk in chunks
    ]

    vector_store = FAISS.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vector_store


def search_vector_store(vector_store, question, k=5):

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results