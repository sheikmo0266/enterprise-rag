from ingestion.pdf_loader import load_pdf
from ingestion.chunking import split_text
from retrieval.embed import create_embeddings
from retrieval.vector_store import create_vector_store, search_vector_store
from backend.gemini import generate_answer

text = load_pdf("simple_machine_learning_10_pages.pdf")

chunks = split_text(text)

embeddings = create_embeddings()

vector_store = create_vector_store(chunks, embeddings)

question = "What is supervised learning?"


results = search_vector_store(vector_store, question)

print("\nQuestion:", question)

print("\nRetrieved chunks:")

for result in results:
    print("\n---")
    print(result.page_content)