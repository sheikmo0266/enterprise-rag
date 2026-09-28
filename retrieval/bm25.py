from rank_bm25 import BM25Okapi


def create_bm25(chunks):

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    tokenized_documents = [
        document.split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25


def search_bm25(bm25, chunks, question, k=5):

    tokenized_question = question.split()

    scores = bm25.get_scores(tokenized_question)

    ranked_indices = scores.argsort()[::-1][:k]

    results = [
        chunks[i]
        for i in ranked_indices
    ]

    return results