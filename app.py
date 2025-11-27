import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


@st.cache_resource
def load_texts():
    texts = json.load(open("texts.json"))
    return [t.lower() for t in texts]  

@st.cache_resource
def load_embeddings():
    return np.load("embeddings.npy")

texts = load_texts()
embeddings = load_embeddings()

def clean_text(q):
    q = q.lower()
    q = q.replace("'", "").replace("’", "")
    return q

def retrieve_transactions(query, embeddings, texts, top_k=5):
    from sentence_transformers import SentenceTransformer

    query = clean_text(query) 

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = embedder.encode([query])

    scores = cosine_similarity(embeddings, query_vec).flatten()
    idx = scores.argsort()[-top_k:][::-1]
    return [texts[i] for i in idx]

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

SYSTEM_RULES = """ You are a helpful assistant. 
Rules: 
1. If the user's message is a greeting, small talk, or general conversation (e.g., hi, hello, how can I assist you), respond normally without using transaction context. 
2. If the user's question is about customers, spending, dates, amounts, products, purchases, etc., use ONLY the retrieved context. 
3. If calculation is required (e.g., "What is Amit’s total spending?"), give a little bit info and concise final answer. 
4. Do NOT guess or invent information. 
5. If the required information is missing, say: "I don't have data for that." """


def generate_answer(query, chat_history):

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

    response = llm.predict(prompt)
    return response.strip()

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
