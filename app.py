import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

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

@st.cache_data(show_spinner=False)
def load_texts():
    with open("texts.json", "r") as f:
        return [t.lower() for t in json.load(f)]

@st.cache_data(show_spinner=False)
def load_embeddings():
    return np.load("embeddings.npy")

encoder = init_encoder()
llm = init_llm()
texts = load_texts()
embeddings = load_embeddings()

def clean_text(q):
    return q.lower().replace("'", "").replace('"', "")


def retrieve_transactions(query, top_k=5):

    if embeddings.size == 0:
        return ["No embeddings found"]

    query = clean_text(query)
    qvec = encoder.encode([query], convert_to_numpy=True)[0]
    qvec = qvec / np.linalg.norm(qvec)

    scores = embeddings @ qvec
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

    context_docs = retrieve_transactions(query, top_k=5)
    context = "\n".join(context_docs)

    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['bot']}"
        for h in chat_history[-2:]
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
