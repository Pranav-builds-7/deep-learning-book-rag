# Deep Learning Book RAG

A Retrieval-Augmented Generation application that answers questions using content retrieved from a deep-learning book.

## Features

- PDF document ingestion
- Semantic search using Hugging Face embeddings
- Chroma vector database
- Mistral-powered question answering
- Streamlit user interface

## Setup

1. Create a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Create a `.env` file:

   MISTRAL_API_KEY=your_api_key

4. Build the vector database if necessary.
5. Run the application:

   streamlit run app.py