import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import faiss
from threading import Thread


st.set_page_config(
    page_title="Transaction RAG Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

load_dotenv()


_ENCODER = None
_FAISS_INDEX = None
_TEXTS = None

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
MODEL_DIR = "./models/paraphrase-MiniLM-L3-v2"


def ensure_model_dir():
    """Guarantee embeddings model exists locally for offline usage."""
    if os.path.exists(MODEL_DIR):
        return True

    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    try:
        with st.spinner("Downloading embedding model..."):
            SentenceTransformer(MODEL_NAME).save(MODEL_DIR)
        return True
    except Exception as exc:
        st.error(f" Failed downloading encoder: {exc}")
        return False

def get_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER

    if not ensure_model_dir():
        return None

    try:
        _ENCODER = SentenceTransformer(MODEL_DIR)
    except Exception as exc:
        st.error(f" Failed loading encoder: {exc}")
        _ENCODER = None

    return _ENCODER

def load_faiss_index():
    global _FAISS_INDEX, _TEXTS
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX, _TEXTS

    try:
        with open("texts.json", "r") as f:
            _TEXTS = [t.lower() for t in json.load(f)]

        emb = np.load("embeddings_fp16.npy", mmap_mode="r").astype(np.float32)
        dim = emb.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        _FAISS_INDEX = index
        return _FAISS_INDEX, _TEXTS

    except Exception as e:
        st.error(f" Failed loading FAISS index: {e}")
        return None, []


@st.cache_resource
def init_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(" GROQ_API_KEY missing.")
        return None

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )

        # Warmup so first request is fast
        def warm():
            try:
                llm.invoke("Ready?")
            except:
                pass

        Thread(target=warm, daemon=True).start()
        return llm

    except Exception as exc:
        st.error(f" LLM init failed: {exc}")
        return None


def clean(q):
    return q.lower().replace("'", "").replace('"', "")


def retrieve(query, index, texts, top_k=3):
    enc = get_encoder()
    if enc is None:
        return ["Embedding model unavailable."]

    q = clean(query)
    qvec = enc.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    qvec = qvec.astype(np.float32)

    scores, idx = index.search(qvec, top_k)
    idx = idx[0]

    return [texts[i] for i in idx]


SYSTEM_RULES = """
You are a helpful assistant.
Rules:
1. Greetings → normal reply.
2. Spending questions → use ONLY context.
3. No fabricated info.
"""

def generate_answer(query):
    index, texts = load_faiss_index()
    if index is None:
        return " Could not load transaction data."

    llm = init_llm()
    if llm is None:
        return " LLM not available."

    docs = retrieve(query, index, texts)
    context = "\n".join(docs)

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
        return f" LLM Error: {e}"


st.title("Transaction RAG Chatbot ⚡")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask about your transactions...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        ans = generate_answer(query)
    st.session_state.messages.append({"role": "assistant", "content": ans})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

