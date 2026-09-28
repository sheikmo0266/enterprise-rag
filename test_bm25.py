from ingestion.pdf_loader import load_pdfs
from ingestion.chunking import split_documents

from retrieval.embed import create_embeddings
from retrieval.vector_store import create_vector_store, search_vector_store

from retrieval.bm25 import create_bm25, search_bm25
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import create_reranker, rerank_results
from backend.gemini import generate_answer
from backend.memory import ConversationMemory

memory = ConversationMemory()



# 1. Load PDFs
documents = load_pdfs("documents")

# 2. Create chunks
chunks = split_documents(documents)

# 3. Create embeddings
embeddings = create_embeddings()

# 4. Create FAISS vector store
vector_store = create_vector_store(
    chunks,
    embeddings
)

# 5. Create BM25
bm25 = create_bm25(chunks)

# 6. Ask a question
question = "What is machine learning?"
memory.add_message(
    "user",
    question
)


# 7. FAISS search
vector_results = search_vector_store(
    vector_store,
    question,
    k=5
)

# 8. BM25 search
bm25_results = search_bm25(
    bm25,
    chunks,
    question,
    k=5
)

# 9. Hybrid search
results = hybrid_search(
    vector_results,
    bm25_results,
    top_k=10
)
print("\nHybrid candidates BEFORE reranking:")

for result in results:

    print(
        f"- Page {result.metadata['page']}: "
        f"{result.metadata['source']}"
    )

# 10. Reranker
reranker = create_reranker()

results = rerank_results(
    reranker,
    question,
    results,
    top_k=10
)
# 11. Create context
context = "\n\n".join(
    result.page_content
    for result in results
)


answer = generate_answer(
    question,
    results,
    memory.get_history()
)
memory.add_message(
    "assistant",
    answer
)

# 13. Display answer
print("\nQuestion:")
print(question)

print("\nGemini Answer:")
print(answer)

question = "What was the percentage increase in gross profit from FY2024 to FY2025"
memory.add_message(
    "user",
    question
)

# 13. Search for second question
vector_results = search_vector_store(
    vector_store,
    question,
    k=5
)

bm25_results = search_bm25(
    bm25,
    chunks,
    question,
    k=5
)

results = hybrid_search(
    vector_results,
    bm25_results,
    top_k=10
)

results = rerank_results(
    reranker,
    question,
    results,
    top_k=10
)

# 14. Second Gemini answer
answer = generate_answer(
    question,
    results,
    memory.get_history()
)

memory.add_message(
    "assistant",
    answer
)

print("\nSecond Question:")
print(question)

print("\nSecond Gemini Answer:")
print(answer)


# 15. Conversation memory
print("\nConversation Memory:")

for message in memory.get_history():

    print(
        f"{message['role']}: "
        f"{message['content']}"
    )



