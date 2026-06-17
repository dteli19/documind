# 📄 DocuMind — AI Document Q&A

An AI-powered document Q&A system that lets users upload any PDF and ask questions in plain English — returning precise answers with source citations from the document.

🚀 **[Live Demo](https://esampgfnwnefwhltwq7te3.streamlit.app/)**

---

## 🎯 Business Problem

Medical affairs teams, pharma reps, and analysts spend hours manually reading dense PDFs — drug labels, clinical trial reports, contracts — to find specific information. DocuMind eliminates that friction by letting anyone ask a question and get an instant, cited answer from the document.

---

## 🏗️ How It Works (RAG Pipeline)

RAG stands for **Retrieval Augmented Generation** — a technique that grounds LLM answers in specific documents rather than general training data.

```
PDF Upload
    ↓
Extract text (PyPDF)
    ↓
Split into chunks (LangChain RecursiveCharacterTextSplitter)
    ↓
Embed chunks → vectors (sentence-transformers: all-MiniLM-L6-v2)
    ↓
Store in FAISS vector database
    ↓
User asks a question
    ↓
Embed question → find 4 most similar chunks (FAISS similarity search)
    ↓
Send chunks + question to Groq Llama 3
    ↓
Answer with source citations
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI | Streamlit | Chat interface and PDF upload |
| PDF Extraction | PyPDF | Extract text from any PDF |
| Chunking | LangChain RecursiveCharacterTextSplitter | Split text into overlapping chunks |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Convert text to vectors |
| Vector Store | FAISS | Fast similarity search |
| LLM | Groq (Llama 3) | Answer generation |
| Orchestration | LangChain RetrievalQA | RAG chain management |

---

## 📁 Project Structure

```
documind/
├── app.py               ← Streamlit UI
├── pdf_processor.py     ← PDF extraction and chunking
├── embeddings.py        ← Vector embeddings and FAISS store
├── qa_chain.py          ← RAG chain connecting retriever + LLM
├── prompts.py           ← LLM prompt template
├── sample_docs/         ← Pre-loaded sample PDF
│   └── sample.pdf       ← GAMIFANT FDA drug label
├── requirements.txt     ← Python dependencies
├── .env                 ← API keys (not committed)
└── .gitignore
```

---

## 🚀 Setup and Run

```bash
git clone https://github.com/dteli19/documind.git
cd documind
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_groq_key_here
```

Run:
```bash
streamlit run app.py
```

---

## 💬 Example Questions

| Document | Question |
|----------|---------|
| Drug label | What are the side effects? |
| Clinical trial | What was the primary endpoint? |
| Contract | What is the termination clause? |
| Annual report | What was Q3 revenue? |

