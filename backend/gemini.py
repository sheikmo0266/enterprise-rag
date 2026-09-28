import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(question, results, history):

    context_parts = []

    for result in results:

        source = result.metadata["source"]
        page = result.metadata["page"]
        text = result.page_content

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    history_text = "\n".join(
    f"{message['role']}: {message['content']}"
    for message in history
)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the information provided in the context.
If the question requires a calculation, use the numerical values from the context and perform the calculation yourself. Show the calculation and give the final result.
Use the source and page information provided in the context.
Do not invent or modify source names or page numbers.
Only cite sources that appear in the provided context.

If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer with the explanation first.

Then provide:
Sources:
- filename, Page number
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text