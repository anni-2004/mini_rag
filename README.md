# Indecimal Support Chatbot (Mini RAG)

This project implements a Retrieval-Augmented Generation (RAG) assistant for the Indecimal construction marketplace. It answers user questions strictly based on the provided internal policies and specifications.

## 🚀 How to Run Locally

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit Chatbot App:
   ```bash
   streamlit run app.py
   ```
   *The app will be accessible at `http://localhost:8501`.

## 🏗️ Architecture & Implementation Notes

### 1. Document Chunking & Embedding
- **Chunking Strategy:** The documents were converted to Markdown/text and chunked sequentially by paragraph (`\n\n` split). Paragraphs under 30 characters are filtered out. This preserves the natural structural boundaries of the content.
- **Embedding Model:** `all-MiniLM-L6-v2` (via `SentenceTransformers`).
  - **Why?** It's a completely open-source, highly efficient model that computes 384-dimensional embeddings very fast on standard CPUs while offering excellent semantic matching accuracy.

### 2. Vector Indexing and Retrieval
- **Vector Store:** `faiss-cpu` (Facebook AI Similarity Search). 
- **Retrieval:** Using `faiss.IndexFlatL2` to perform exact nearest neighbor search based on L2 distance. We retrieve the `top_k=5` most relevant paragraphs for every user query.

### 3. Model Choice & Grounding
Users can toggle between two LLM execution modes from the sidebar:
1. **OpenRouter (Free API):** Uses an external powerful free model (e.g. `google/gemini-2.5-flash:free`). It is fast and fluent.
2. **Local Open-Source LLM [Bonus Feature]:** Uses `Qwen/Qwen2.5-0.5B-Instruct` (or a similar ~0.5B-1.5B parameters model) through Hugging Face `transformers` running 100% locally.

**How is grounding enforced?**
The AI is provided with a strict `System Prompt`:
*"You are a helpful AI assistant for Indecimal... Answer the user's question explicitly and strictly based on the provided Context. If the answer cannot be deduced from the Context, kindly state that you do not have enough information."*

The prompt actively isolates the LLM's world knowledge, constraining it strictly to the retrieved strings injected dynamically into the `Context` block.

### 4. Transparency
The custom Streamlit frontend includes an expandable "Show Retrieved Context" element for every AI response. This allows users to independently verify exactly which document chunks were retrieved to author the generated answer.

---

## 🧪 Evaluation & Quality Analysis

### Test Questions Derived from Documents
1. What is the per sqft rate for the Premier package?
2. Does Indecimal provide real-time project tracking?
3. What is the wallet allowance for a Main Door in the Essential package?
4. What brands of cement are used for the Pinnacle package?
5. Are there any penalties for project delays?
6. Who is the target audience for the Company Overview document?
7. What paints are used for Exterior Painting in the Infinia package?
8. How does Indecimal handle payments to contractors?
9. Is there any warranty provided post-handover?
10. What is the standard ceiling height across all packages?

### Quality Analysis Observations
- **Relevance of Retrieved Chunks:** The FAISS semantic retrieval hits highly accurate chunks. Queries asking about specific policies usually yield the matching structural header directly.
- **Presence of Hallucinations:** When prompted queries outside the documents (e.g., "What is the capital of France?"), the strict systemic prompt successfully restricts the model to reply "I do not have enough information".
- **Local vs OpenRouter Comparison:** 
  - *Answer Quality:* OpenRouter endpoints yield highly conversational, fully rounded sentences. Local Open Source models (like 0.5B parameters) are more concise and utilitarian but successfully retrieve facts.
  - *Latency:* OpenRouter delivers responses in ~1-2 seconds. Local CPU inference can take 10-30 seconds depending on the hardware. 
  - *Groundedness:* Both effectively stay within the bounds of the provided context.
