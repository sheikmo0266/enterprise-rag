from langchain_core.documents import Document


def hybrid_search(
    vector_results,
    bm25_results,
    top_k=5
):

    combined = {}

    # Add vector search results
    for rank, result in enumerate(vector_results):

        key = result.page_content

        if key not in combined:

            combined[key] = {
                "document": result,
                "score": 0
            }

        combined[key]["score"] += 1 / (60 + rank + 1)

    # Add BM25 results
    for rank, result in enumerate(bm25_results):

        key = result["text"]

        if key not in combined:

            document = Document(
                page_content=result["text"],
                metadata={
                    "page": result["page"],
                    "source": result["source"]
                }
            )

            combined[key] = {
                "document": document,
                "score": 0
            }

        combined[key]["score"] += 1 / (60 + rank + 1)

    # Sort by RRF score
    ranked_results = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Return top results
    return [
        item["document"]
        for item in ranked_results[:top_k]
    ]