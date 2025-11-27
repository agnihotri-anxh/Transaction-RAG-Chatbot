import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Transaction RAG Chatbot")

@st.cache_data
def load_texts():
    try:
        with open("texts.json", "r") as f:
            texts = json.load(f)
        return [t.lower() for t in texts]
    except FileNotFoundError:
        st.error("Error: 'texts.json' not found.")
        return []

@st.cache_data
def load_embeddings():
    try:
        return np.load("embeddings.npy", mmap_mode="r")
    except FileNotFoundError:
        st.error("Error: 'embeddings.npy' not found.")
        return np.array([])
    except ValueError:
        st.error("Error: Could not load embeddings. File might be corrupt.")
        return np.array([])

@st.cache_resource(show_spinner=False)
def load_query_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment variables.")
        return None
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
    )

def clean_text(q):
    q = q.lower()
    q = q.replace("'", "").replace('"', "")
    return q

def retrieve_transactions(query, embeddings, texts, top_k=5):
    if embeddings.size == 0 or not texts:
        return ["No transaction data loaded."]
        
    query = clean_text(query)
    encoder = load_query_encoder()  
    query_vec = encoder.encode([query], convert_to_numpy=True)[0]

    query_norm = query_vec / np.linalg.norm(query_vec)
    scores = np.dot(embeddings, query_norm)

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    scores = scores / (embedding_norms + 1e-8)  
    
    idx = scores.argsort()[-top_k:][::-1]
    
    return [texts[i] for i in idx]

SYSTEM_RULES = """You are a helpful assistant.
Rules:
1. Greetings -> respond normally(like, how can I assist you).
2. Questions about spending -> use ONLY context.
3. Calculations -> concise final answer.
4. Do NOT invent information.
5. If missing -> say: 'I don't have data for that.'"""

def generate_answer(query, chat_history):
    texts = load_texts()
    embeddings = load_embeddings()
    llm = load_llm()

    if not texts or embeddings.size == 0 or not llm:
        return "I'm having trouble loading the transaction data or the language model. Please check the file paths and API key."

    recent_history = chat_history[-2:]
    
    context_docs = retrieve_transactions(query, embeddings, texts, top_k=5)
    context = "\n".join(context_docs)

    history_text = "\n".join([
        f"User: {h['user']}\nAssistant: {h['bot']}"
        for h in recent_history
    ])

    prompt = f"""
{SYSTEM_RULES}

Chat History:
{history_text}

Context:
{context}

Question: {query}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"An error occurred while calling the LLM: {e}"

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