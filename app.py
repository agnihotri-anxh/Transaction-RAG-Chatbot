import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Transaction RAG Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

load_dotenv()

_ENCODER = None


def get_encoder():
    """Load the encoder once per process (no Streamlit cache)."""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER

    model_path = "./models/paraphrase-MiniLM-L3-v2"
    if not os.path.exists(model_path):
        st.error("Embedding model folder not found.")
        return None

    try:
        _ENCODER = SentenceTransformer(model_path)
    except Exception as exc:
        st.error(f"Failed to load encoder: {exc}")
        _ENCODER = None
    return _ENCODER

from threading import Thread


@st.cache_resource
def init_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set in environment.")
        return None

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
        )

        # Warm up Groq asynchronously so first user query isn't blocked
        def _warm_up():
            try:
                llm.invoke("Say 'ready'.")
            except Exception:
                pass

        Thread(target=_warm_up, daemon=True).start()
        return llm
    except Exception as exc:
        st.error(f"Failed to initialize LLM: {exc}")
        return None

@st.cache_resource
def load_texts():
    try:
        with open("texts.json", "r") as f:
            return [t.lower() for t in json.load(f)]
    except:
        return []

@st.cache_resource
def load_embeddings():
    try:
        emb = np.load("embeddings.npy", mmap_mode="r")
        return np.asarray(emb, dtype=np.float32)
    except:
        return np.array([])

def clean_text(q):
    return q.lower().replace("'", "").replace('"', "")

def retrieve_transactions(query, embeddings, texts, top_k=3):
    if embeddings.size == 0 or not texts:
        return ["No transaction data loaded."]
    encoder = get_encoder()
    if encoder is None:
        return ["Embedding model unavailable."]
    query = clean_text(query)
    qvec = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    qvec = qvec.astype(np.float32, copy=False)
    scores = embeddings @ qvec
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [texts[i] for i in top_idx]

SYSTEM_RULES = """
You are a helpful assistant.
Rules:
1. Greetings → normal reply(like Greeting, how can I assist you).
2. Spending questions → use ONLY context.
3. No fabricated info.
"""

def generate_answer(query):
    texts = load_texts()
    embeddings = load_embeddings()
    llm = init_llm()
    if llm is None:
        return "LLM not available. Check GROQ_API_KEY."
    context_docs = retrieve_transactions(query, embeddings, texts)
    context = "\n".join(context_docs)
    prompt = f"""
{SYSTEM_RULES}

Context:
{context}

Question: {query}
Answer:
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"LLM error: {e}"

st.title("Transaction RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask something about the transactions...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        answer = generate_answer(query)
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
