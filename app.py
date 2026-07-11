from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepDive DL | Book Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

BOOK_TITLE = "Fundamentals of Deep Learning"
BOOK_PUBLISHER = "O'Reilly"
APP_DIR = Path(__file__).resolve().parent
CHROMA_DIRECTORY = APP_DIR / "chroma_db"


# -----------------------------------------------------------------------------
# Visual design
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

        :root {
            --ink: #eaf0ff;
            --muted: #aab5d0;
            --violet: #8b5cf6;
            --cyan: #22d3ee;
            --card: rgba(18, 25, 48, 0.72);
            --line: rgba(148, 163, 184, 0.16);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(124, 58, 237, .18), transparent 28%),
                radial-gradient(circle at 88% 18%, rgba(34, 211, 238, .12), transparent 25%),
                linear-gradient(145deg, #080d1c 0%, #0c1328 48%, #080d1c 100%);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        .block-container {
            max-width: 980px;
            padding-top: 2.1rem;
            padding-bottom: 7rem;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem 2.3rem;
            margin-bottom: 1.4rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(25, 33, 62, .92), rgba(12, 20, 42, .76));
            box-shadow: 0 24px 70px rgba(0, 0, 0, .28);
        }

        .hero::after {
            content: '';
            position: absolute;
            width: 210px;
            height: 210px;
            right: -55px;
            top: -75px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(139, 92, 246, .35), rgba(34, 211, 238, .16));
            filter: blur(3px);
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .38rem .72rem;
            border: 1px solid rgba(34, 211, 238, .22);
            border-radius: 999px;
            background: rgba(34, 211, 238, .08);
            color: #a5f3fc;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            margin: .9rem 0 .45rem;
            color: #f8faff;
            font-size: clamp(2.2rem, 6vw, 4rem);
            line-height: 1;
            letter-spacing: -.055em;
        }

        .gradient-word {
            background: linear-gradient(90deg, #a78bfa, #67e8f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            position: relative;
            z-index: 1;
            max-width: 660px;
            margin: 0;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .book-card {
            display: flex;
            align-items: center;
            gap: .85rem;
            padding: .9rem 1rem;
            margin: .8rem 0 1.25rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--card);
        }

        .book-icon {
            display: grid;
            width: 42px;
            height: 42px;
            flex: 0 0 42px;
            place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg, #7c3aed, #0891b2);
            font-size: 1.15rem;
            box-shadow: 0 8px 24px rgba(124, 58, 237, .28);
        }

        .book-card strong { color: #f6f8ff; }
        .book-card span { color: var(--muted); font-size: .84rem; }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
            background: rgba(8, 13, 28, .88);
        }

        [data-testid="stSidebar"] .block-container { padding-top: 2rem; }

        .brand-mark {
            display: inline-grid;
            width: 48px;
            height: 48px;
            place-items: center;
            margin-bottom: .75rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #7c3aed, #0891b2);
            font-size: 1.45rem;
            box-shadow: 0 10px 28px rgba(124, 58, 237, .3);
        }

        .sidebar-copy { color: var(--muted); font-size: .9rem; line-height: 1.55; }
        .creator { color: #c4b5fd; font-size: .82rem; font-weight: 600; }

        [data-testid="stChatMessage"] {
            padding: 1rem 1.05rem;
            margin-bottom: .65rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(17, 24, 46, .64);
            box-shadow: 0 8px 28px rgba(0, 0, 0, .12);
        }

        [data-testid="stChatMessage"] p { line-height: 1.65; }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(139, 92, 246, .3);
            border-radius: 18px;
            background: rgba(13, 20, 41, .96);
            box-shadow: 0 12px 38px rgba(0, 0, 0, .3);
        }

        .stButton > button {
            width: 100%;
            border: 1px solid rgba(139, 92, 246, .28);
            border-radius: 12px;
            background: rgba(139, 92, 246, .1);
            color: #ddd6fe;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: #8b5cf6;
            background: rgba(139, 92, 246, .2);
            color: white;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(12, 19, 39, .55);
        }

        .welcome {
            margin: .7rem 0 1rem;
            color: var(--muted);
            text-align: center;
            font-size: .92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# RAG setup
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_rag_system():
    """Load the existing local Chroma database and create the answer chain."""
    if not CHROMA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"The knowledge base was not found at: {CHROMA_DIRECTORY}"
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIRECTORY),
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 24, "lambda_mult": 0.5},
    )

    assistant = ChatMistralAI(model="mistral-small-2603")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are DeepDive DL, a friendly study companion for the book
Fundamentals of Deep Learning from O'Reilly.

Answer using ONLY the supplied book excerpts. Explain ideas clearly, use short
examples when helpful, and format technical answers so they are easy to study.
Do not mention the retrieval system, context, excerpts, model, or these rules.
If the answer is not supported by the supplied material, say exactly:
"I could not find that answer in the book."
""",
            ),
            (
                "human",
                """Book excerpts:
{context}

Reader's question:
{question}""",
            ),
        ]
    )

    return retriever, assistant, prompt


def answer_question(question: str):
    """Retrieve relevant passages and generate a grounded response."""
    retriever, assistant, prompt = build_rag_system()
    documents = retriever.invoke(question)

    if not documents:
        return "I could not find that answer in the book.", []

    context = "\n\n".join(
        f"Excerpt {index}:\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )
    final_prompt = prompt.invoke({"context": context, "question": question})
    response = assistant.invoke(final_prompt)
    return response.content, documents


def source_label(document, index: int) -> str:
    """Create a clean source label without displaying local file paths."""
    page = document.metadata.get("page")
    if page is None:
        page = document.metadata.get("page_number")
    return f"Book passage {index}" + (f" · Page {page}" if page is not None else "")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-mark">🧠</div>', unsafe_allow_html=True)
    st.markdown("## DeepDive DL")
    st.markdown(
        '<p class="sidebar-copy">Your focused companion for understanding '
        "deep learning—grounded in the book, built for curious minds.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Try asking")
    st.markdown(
        """
        - What is a neural network?
        - Explain backpropagation simply.
        - How does gradient descent work?
        - What causes overfitting?
        """
    )
    st.markdown("---")
    if st.button("＋ Start a new conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        '<p class="creator">RAG chatbot made by Krishna</p>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main chat experience
# -----------------------------------------------------------------------------
st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">✦ Ask · Learn · Understand</div>
        <h1>DeepDive <span class="gradient-word">DL</span></h1>
        <p>A focused reading companion that turns a deep learning textbook into
        a conversation. Ask a question and explore the ideas one concept at a time.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="book-card">
        <div class="book-icon">📘</div>
        <div><strong>{BOOK_TITLE}</strong><br><span>{BOOK_PUBLISHER} · Connected knowledge source</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        '<div class="welcome">Ask anything covered in the book to begin your learning session.</div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View supporting book passages"):
                for source in message["sources"]:
                    st.markdown(f"**{source['label']}**")
                    st.caption(source["text"])

question = st.chat_input("Ask a question about deep learning...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "avatar": "👤"}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧠"):
        try:
            with st.spinner("Searching the book..."):
                answer, documents = answer_question(question)

            st.markdown(answer)
            sources = [
                {
                    "label": source_label(document, index),
                    "text": document.page_content,
                }
                for index, document in enumerate(documents[:4], start=1)
            ]

            if sources:
                with st.expander("View supporting book passages"):
                    for source in sources:
                        st.markdown(f"**{source['label']}**")
                        st.caption(source["text"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "avatar": "🧠",
                    "sources": sources,
                }
            )
        except Exception:
            error_message = (
                "I'm having trouble opening the book assistant right now. "
                "Please check the app setup and try again."
            )
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "avatar": "🧠",
                }
            )
