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
def load_texts():
    texts = json.load(open("texts.json"))
    return [t.lower() for t in texts]

@st.cache_resource
def load_embeddings():
    return np.load("embeddings.npy", mmap_mode="r")

@st.cache_resource(show_spinner=False)
def load_query_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
    )

def clean_text(q):
    q = q.lower()
    q = q.replace("'", "").replace("'", "")
    return q

def retrieve_transactions(query, embeddings, texts, top_k=5):
    from sklearn.metrics.pairwise import cosine_similarity

    query = clean_text(query)
    encoder = load_query_encoder()   
    query_vec = encoder.encode([query])
    scores = cosine_similarity(embeddings, query_vec).flatten()
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
    # Lazy load only when first query arrives
    texts = load_texts()
    embeddings = load_embeddings()
    llm = load_llm()

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

    answer = generate_answer(query, st.session_state.history)
    st.session_state.history.append({"user": query, "bot": answer})

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
