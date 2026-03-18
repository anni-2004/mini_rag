import streamlit as st
import os
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import torch

# --- CONFIGURATION ---
DATA_DIR = "./"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Small/fast for demonstration

st.set_page_config(page_title="Indecimal Mini RAG", page_icon="🏗️", layout="wide")

st.title("🏗️ Indecimal Support Chatbot (Mini RAG)")
st.markdown("This AI assistant answers construction inquiries strictly based on internal Indecimal documents.")

# --- INITIALIZATION ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_local_llm():
    # Load model and tokenizer only once if local mode is selected
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_LLM_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, 
                    max_new_tokens=256, temperature=0.1, top_p=0.9,
                    device=0 if torch.cuda.is_available() else -1)

@st.cache_data
def process_documents():
    docs = []
    # Read doc_1.md, doc_2.md, doc_3.md
    for filename in ["doc_1.md", "doc_2.md", "doc_3.md"]:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple chunking: split by paragraphs
                paragraphs = content.split('\n\n')
                for p in paragraphs:
                    p = p.strip()
                    if len(p) > 30: # ignore very short chunks
                        docs.append({"source": filename, "text": p})
    return docs

@st.cache_resource
def build_faiss_index(docs):
    embedder = load_embedder()
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, texts, docs

# Load components
embedder = load_embedder()
docs = process_documents()
if len(docs) > 0:
    index, chunk_texts, chunk_metadata = build_faiss_index(docs)
else:
    st.error("Document files are missing or empty!")
    st.stop()

# --- SIDEBAR CONFIG ---
st.sidebar.header("Configuration")
llm_choice = st.sidebar.radio("LLM Provider", ["OpenRouter (Free API)", "Local (Hugging Face)"])
openrouter_key = ""
openrouter_model = "google/gemini-2.5-flash:free"
if llm_choice == "OpenRouter (Free API)":
    openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")
    st.sidebar.markdown("[Get Free Key](https://openrouter.ai/keys)")
    st.sidebar.info("Recommmended Model: `google/gemini-2.5-flash:free` or `mistralai/mistral-7b-instruct:free`")

# --- RAG RETRIEVAL ---
def retrieve(query, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    retrieved = []
    for idx in indices[0]:
        retrieved.append(chunk_metadata[idx])
    return retrieved

# --- STREAMLIT CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            with st.expander("Show Retrieved Context"):
                for i, ctx in enumerate(msg["contexts"]):
                    st.markdown(f"**Chunk {i+1}** *(Source: {ctx['source']})*")
                    st.text(ctx["text"])

prompt = st.chat_input("Ask about Indecimal construction policies...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    start_time = time.time()
    # 1. Retrieval
    retrieved_chunks = retrieve(prompt, top_k=3)
    context_text = "\n\n---\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in retrieved_chunks])
    
    # 2. Generation
    system_prompt = (
        "You are a helpful AI assistant for Indecimal, a construction marketplace. "
        "Answer the user's question explicitly and strictly based on the provided Context. "
        "If the answer cannot be deduced from the Context, kindly state that you do not have enough information."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}"

    final_answer = ""
    with st.spinner("Generating answer..."):
        if llm_choice == "OpenRouter (Free API)":
            if not openrouter_key:
                st.error("Please enter your OpenRouter API Key in the sidebar.")
                st.stop()
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
                response = client.chat.completions.create(
                    model=openrouter_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                )
                final_answer = response.choices[0].message.content
            except Exception as e:
                final_answer = f"Error calling OpenRouterAPI: {str(e)}"
        else:
            # Local LLM generator
            local_pipe = load_local_llm()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = local_pipe(messages, max_new_tokens=256, return_full_text=False)
            final_answer = response[0]['generated_text']

    latency = time.time() - start_time

    # 3. Output display
    with st.chat_message("assistant"):
        st.write(final_answer)
        st.caption(f"⏱️ Latency: {latency:.2f}s | Provider: {llm_choice}")
        with st.expander("Show Retrieved Context (Transparency)"):
            for i, ctx in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}** *(Source: {ctx['source']})*")
                st.text(ctx["text"])
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_answer, 
        "contexts": retrieved_chunks
    })
