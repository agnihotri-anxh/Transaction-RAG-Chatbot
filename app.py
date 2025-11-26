import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

@st.cache_resource
def load_local_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_local_embedding_model()

def get_embeddings(text_list):
    """Generate embeddings locally (fast, offline, reliable)."""
    return embedder.encode(text_list)

@st.cache_resource
def load_transactions(file_path="transactions.json"):
    with open(file_path, "r") as f:
        return json.load(f)

transactions = load_transactions()

texts = [
    f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
    for t in transactions
]
@st.cache_resource
def load_all_embeddings():
    return get_embeddings(texts)

embeddings = load_all_embeddings()
def retrieve_transactions(query, embeddings, texts, top_k=5):
    query_vec = get_embeddings([query])
    scores = cosine_similarity(embeddings, query_vec).flatten()
    idx = scores.argsort()[-top_k:][::-1]
    return [texts[i] for i in idx]
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

SYSTEM_RULES = """ You are a helpful assistant. 
Rules: 
1. If the user's message is a greeting, small talk, or general conversation (e.g., hi, hello, how can I assist you), respond normally without using transaction context. 
2. If the user's question is about customers, spending, dates, amounts, products, purchases, etc., use ONLY the retrieved context.
3. If calculation is required (e.g., "What is Amit’s total spending?"), give a concise final answer.
4. Do NOT guess or invent information.
5. If the required information is missing, say: "I don't have data for that." 
"""


def generate_answer(query, chat_history):
    context_docs = retrieve_transactions(query, embeddings, texts, top_k=5)
    context = "\n".join(context_docs)

    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['bot']}" for h in chat_history]
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
st.set_page_config(page_title="Transaction RAG Chatbot")
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
