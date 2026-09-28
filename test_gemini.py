from ingestion.pdf_loader import load_pdfs
from ingestion.chunking import split_documents
from retrieval.embed import create_embeddings
from retrieval.vector_store import create_vector_store, search_vector_store
from retrieval.reranker import create_reranker, rerank_results
from backend.gemini import generate_answer


# 1. Load all PDFs
documents = load_pdfs("documents")


# 2. Split into chunks
chunks = split_documents(documents)


# 3. Create embeddings
embeddings = create_embeddings()


# 4. Create FAISS vector store
vector_store = create_vector_store(
    chunks,
    embeddings
)


# 5. Create reranker
reranker = create_reranker()


# 6. Ask a question
question = "What is a Python list?"


# 7. Retrieve candidate chunks
results = search_vector_store(
    vector_store,
    question,
    k=5
)


# 8. Rerank results
results = rerank_results(
    reranker,
    question,
    results,
    top_k=3
)


# 9. Create context
context = "\n\n".join(
    result.page_content
    for result in results
)


# 10. Generate answer
answer = generate_answer(
    question,
    results,
    []
)

# 11. Display answer
print("\nQuestion:")
print(question)

print("\nGemini Answer:")
print(answer)


# 12. Display sources
print("\nSources:")

for result in results:
    print(
        f"- Page {result.metadata['page']}: "
        f"{result.metadata['source']}"
    )