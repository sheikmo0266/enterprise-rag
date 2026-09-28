from sentence_transformers import CrossEncoder


def create_reranker():
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return reranker


def rerank_results(reranker, question, results, top_k=3):

    pairs = [
        (question, result.page_content)
        for result in results
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        result
        for score, result in ranked_results[:top_k]
    ]