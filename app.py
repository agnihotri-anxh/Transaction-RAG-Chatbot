import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

st.set_page_config(page_title="Transaction RAG Chatbot")
load_dotenv()


@st.cache_resource(show_spinner=False)
def init_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")  

@st.cache_resource(show_spinner=False)
def init_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
    )

@st.cache_resource
def load_texts():
    """Load texts - cached as resource"""
    try:
        with open("texts.json", "r") as f:
            return [t.lower() for t in json.load(f)]
    except FileNotFoundError:
        st.error("Error: 'texts.json' not found.")
        return []

@st.cache_resource
def load_embeddings():
    """Load embeddings with memory mapping - CRITICAL for memory efficiency"""
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
    """Retrieve transactions with memory-efficient similarity calculation"""
    if embeddings.size == 0 or not texts:
        return ["No transaction data loaded."]
    
    query = clean_text(query)
    encoder = init_encoder()

    qvec = encoder.encode([query], convert_to_numpy=True)[0]
    qvec = qvec / (np.linalg.norm(qvec) + 1e-8)

    scores = np.dot(embeddings, qvec)

    if len(embeddings) > 10000:

        idx = scores.argsort()[-top_k:][::-1]
    else:

        norms = np.linalg.norm(embeddings, axis=1)
        scores = scores / (norms + 1e-8)
        idx = scores.argsort()[-top_k:][::-1]

    return [texts[i] for i in idx]

SYSTEM_RULES = """You are a helpful assistant.
Rules:
1. Greetings -> normal reply.
2. Spending questions -> use ONLY context.
3. Calculations -> concise.
4. No invented info.
5. Missing data -> say 'I don't have data for that.'
"""

def generate_answer(query, chat_history):
    """Generate answer with lazy-loaded resources"""
    # Lazy load resources only when needed (saves memory on startup)
    texts = load_texts()
    embeddings = load_embeddings()
    llm = init_llm()

    context_docs = retrieve_transactions(query, embeddings, texts, top_k=5)
    context = "\n".join(context_docs)

    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['bot']}"
        for h in chat_history[-1:]   
    )

    prompt = f"""
{SYSTEM_RULES}

        Chat History:
        {history_text}

        Context:
        {context}

        Question: {query}

        Answer:"""

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
