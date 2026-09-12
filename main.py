from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
)
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
import os

load_dotenv()

# Deep Learning books contain complex technical terminology. 
# BGE-Small remains excellent, but ensuring clean layout vector space mapping is critical.
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma(
    persist_directory="chroma_dbNew",
    embedding_function=embedding_model
)

# Initialize Gemini Flash with strict zero temperature for technical correctness
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0,
)

# Enhanced System Prompt engineered for deep technical texts (O'Reilly style)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI Research Assistant specializing in Deep Learning and Neural Network Architectures. 
Your task is to answer user technical queries using exclusively the indexed book context provided below.

CRITICAL INSTRUCTIONS:
1. GROUNDING: Rely ONLY on the facts, equations, formulas, and code snippets explicitly mentioned in the context. Do not extrapolate, assume, or inject external general AI knowledge.
2. HANDLING GAPS: If the provided context does not contain sufficient technical details to fully resolve the query, state exactly: "I could not find the answer in the document." Do not try to construct an unverified explanation.
3. FORMATTING: Present your technical response cleanly using structural markdown elements:
   - Use structural code blocks (```python) for mathematical operations or network layers.
   - Use bold highlights for core technical terms (e.g., **Backpropagation**, **Gradient Descent**).
   - Use bullet points to break down complex multi-step processes or algorithms.
"""
        ),
        (
            "human",
            """Context from Textbook:
{context}

Technical Question:
{question}
"""
        )
    ]
)

print("🚀 Advanced Deep Learning RAG Engine Created via Gemini 3.6 Flash")
print("Type your query below or press 0 to exit.")

while True:
    query = input("\nYou: ")

    if query == "0":
        break

    # OPTIMIZATION: Maximum Marginal Relevance (MMR) Search
    # This ensures a diverse mix of context (e.g., getting a mix of math formulas, text, 
    # and code chunks rather than retrieving 20 variations of the exact same introductory paragraph).
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=15,         # Leverages Gemini's huge token window safely
        fetch_k=40,   # Evaluates a wider target pool for diversity analysis
        lambda_mult=0.6 # 0.6 strikes the ideal balance between raw relevance and content diversity
    )

    if not docs:
        print("\nAI: I could not find the answer in the document.")
        continue

    # Cleanly format the retrieved book context chunks
    context = "\n\n--- New Section ---\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    try:
        response = llm.invoke(final_prompt)
        answer = response.content
        if isinstance(answer, list):
            answer = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in answer
            )
        print(f"\nAI:\n{answer}")
    except ResourceExhausted:
        print(
            "\nAI: Gemini API rate limit reached. "
            "Please wait a moment before trying again, or check your Google AI Studio quota limits."
        )
    except GoogleInvalidRequestError as error:
        print(
            "\nAI: Gemini rejected the request. Check that GOOGLE_API_KEY in .env "
            f"is valid and enabled for the Generative Language API. ({error})"
        )
    except GoogleModelNotFoundError as error:
        print(f"\nAI: The configured Gemini model is unavailable. ({error})")
    except GoogleAPICallError as error:
        print(f"\nAI: An internal Gemini system error occurred ({error.message}).")
