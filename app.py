import json
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

st.set_page_config(
    page_title="Transaction RAG Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


load_dotenv()

def get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def get_chat_groq():
    from langchain_groq import ChatGroq
    return ChatGroq


@st.cache_resource(show_spinner="Loading embedding model...")
def init_encoder():
    """Lazy load SentenceTransformer only when needed"""
    try:
        SentenceTransformer = get_sentence_transformer()
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        raise  

@st.cache_resource(show_spinner="Initializing LLM...")
def init_llm():
    """Lazy load ChatGroq only when needed"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        ChatGroq = get_chat_groq()
        return ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        raise

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
    try:
        texts = load_texts()
        embeddings = load_embeddings()
        
        # Check if data is loaded
        if not texts or embeddings.size == 0:
            return "Error: Transaction data not loaded. Please check if texts.json and embeddings.npy exist."
        
        try:
            llm = init_llm()
        except Exception as e:
            return f"Error initializing LLM: {str(e)}. Please check GROQ_API_KEY environment variable."

        try:
            context_docs = retrieve_transactions(query, embeddings, texts, top_k=5)
            context = "\n".join(context_docs)
        except Exception as e:
            return f"Error retrieving transactions: {str(e)}"

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

        # Call LLM with error handling
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error calling LLM: {str(e)}. Please check your API key and network connection."
            
    except Exception as e:
        return f"Unexpected error: {str(e)}"


st.title("Transaction RAG Chatbot")

# Show info about first query
if "model_loaded" not in st.session_state:
    st.info("ℹ First query may take 1-2 minutes to download the embedding model (~80MB). Subsequent queries will be faster.")

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

    with st.spinner("Thinking... (First query may take 1-2 minutes to load models)"):
        try:
            answer = generate_answer(query, st.session_state.history)
            st.session_state.model_loaded = True  # Mark model as loaded
        except Exception as e:
            answer = f"Error: {str(e)}. Please check the logs or try again."

    st.session_state.history.append({"user": query, "bot": answer})
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
