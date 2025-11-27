import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Transaction RAG Chatbot")
load_dotenv()

@st.cache_resource
def load_llm():
    """Load LLM model - cached as resource (not serializable)"""
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
    )

@st.cache_resource(show_spinner=False)
def load_query_encoder():
    """Load query encoder - cached as resource (not serializable)"""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_texts():
    """Load texts - cached as resource (file handle)"""
    try:
        with open("texts.json", "r") as f:
            return [t.lower() for t in json.load(f)]
    except FileNotFoundError:
        st.error("Error: 'texts.json' not found.")
        return []

@st.cache_resource
def load_embeddings():
    """Load embeddings with memory mapping - cached as resource (file handle)"""
    try:
        return np.load("embeddings.npy", mmap_mode="r")
    except FileNotFoundError:
        st.error("Error: 'embeddings.npy' not found.")
        return np.array([])
    except ValueError:
        st.error("Error: Could not load embeddings. File might be corrupt.")
        return np.array([])


def clean_text(q):
    return q.lower().replace("'", "").replace('"', "")

def retrieve_transactions(query, embeddings, texts, top_k=5):
    if embeddings.size == 0 or not texts:
        return ["No transaction data loaded."]

    query = clean_text(query)
    encoder = load_query_encoder()
    
    vec = encoder.encode([query], convert_to_numpy=True)[0]
    vec = vec / np.linalg.norm(vec)

    scores = embeddings @ vec
    norms = np.linalg.norm(embeddings, axis=1)
    scores = scores / (norms + 1e-8)

    idx = scores.argsort()[-top_k:][::-1]
    return [texts[i] for i in idx]


SYSTEM_RULES = """
You are a helpful assistant.
Use ONLY context. Do NOT guess. 
If missing, say: "I don't have data for that."
"""


def generate_answer(query, chat_history):
    # Lazy load resources only when needed (on first query)
    texts = load_texts()
    embeddings = load_embeddings()
    llm = load_llm()

    recent_history = chat_history[-2:]

    context_docs = retrieve_transactions(query, embeddings, texts, top_k=5)
    context = "\n".join(context_docs)

    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['bot']}"
        for h in recent_history
    )

    prompt = f"""
{SYSTEM_RULES}

Chat History:
{history_text}

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content.strip()


st.title("Transaction RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something about the transactions...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        answer = generate_answer(query, st.session_state.history)

    st.session_state.history.append({"user": query, "bot": answer})

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)