🚀 Built a Deep Learning Book RAG Assistant 🧠📚

I recently built **DeepDive DL**, an interactive Retrieval-Augmented Generation (RAG) application that turns a deep learning textbook into a conversational study companion.

Instead of asking an LLM to answer purely from its pretrained knowledge, the application first retrieves relevant passages from the book and then uses Google Gemini to generate a grounded response.

🔍 **How it works:**

PDF → Text Chunking → Hugging Face Embeddings → Chroma Vector Database → MMR Retrieval → Google Gemini → Streamlit UI

🛠️ **Tech Stack:**
• Python
• LangChain
• ChromaDB
• Hugging Face Embeddings
• BGE-small-en-v1.5
• Google Gemini
• PyPDF
• Streamlit

✨ **Key features:**
• Semantic search over the book
• MMR-based retrieval for relevant and diverse passages
• Answers grounded in retrieved book content
• Supporting passages available for verification
• Modern Streamlit chat interface
• Local vector database

The biggest takeaway for me was understanding that building an effective RAG application is more than simply connecting an LLM to a vector database.

It involves thinking about **document processing, chunking, embeddings, retrieval strategy, prompt design, grounding, and user experience** as one complete pipeline.

This project helped me strengthen my understanding of how modern AI applications are built end-to-end.

🔗 GitHub: https://github.com/Pranav-builds-7/deep-learning-book-rag

#RAG #GenerativeAI #LLM #LangChain #ChromaDB #GoogleGemini #HuggingFace #Streamlit #Python #AIEngineering #MachineLearning
