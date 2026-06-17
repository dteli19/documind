import streamlit as st
import os
from pdf_processor import process_pdf
from embeddings import create_embeddings, get_retriever
from qa_chain import create_qa_chain, get_answer

st.set_page_config(
    page_title="DocuMind - AI Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; color: white; }
    .main-header p { font-size: 1rem; color: #a0aec0; margin: 0.5rem 0 0 0; }
    .source-pill {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .source-text {
        background: #f8fafc;
        border-left: 3px solid #0f3460;
        padding: 0.6rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #374151;
        margin: 0.3rem 0 0.8rem 0;
        font-style: italic;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📄 DocuMind — AI Document Q&A</h1>
    <p>Upload any PDF and ask questions in plain English · Powered by RAG · Groq Llama 3 · FAISS</p>
</div>
""", unsafe_allow_html=True)

SAMPLE_PDF_PATH = "sample_docs/sample.pdf"
SAMPLE_PDF_NAME = "GAMIFANT Drug Label (FDA)"


def process_and_store(pdf_path: str, pdf_name: str):
    with st.spinner(f"📖 Reading {pdf_name}..."):
        chunks = process_pdf(pdf_path)
        st.info(f"✅ Created {len(chunks)} text chunks")

    with st.spinner("🔢 Building vector store..."):
        vector_store = create_embeddings(chunks)
        retriever = get_retriever(vector_store)
        qa_chain = create_qa_chain(retriever)

    st.session_state.qa_chain = qa_chain
    st.session_state.pdf_name = pdf_name
    st.session_state.chunks_count = len(chunks)
    st.session_state.messages = []
    st.success(f"Ready! Ask questions about {pdf_name}")


with st.sidebar:
    st.markdown("### 📂 Document Selection")
    if st.button("🧪 Load Sample Pharma PDF", use_container_width=True):
        if os.path.exists(SAMPLE_PDF_PATH):
            process_and_store(SAMPLE_PDF_PATH, SAMPLE_PDF_NAME)
        else:
            st.error("Sample PDF not found in sample_docs/")

    st.markdown("---")
    st.markdown("**Or upload your own PDF:**")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        process_and_store(temp_path, uploaded_file.name)

    if "pdf_name" in st.session_state:
        st.markdown("---")
        st.markdown("**📊 Loaded Document**")
        st.metric("Chunks indexed", st.session_state.chunks_count)

    st.markdown("---")
    st.markdown("### 💡 Try asking...")
    st.markdown("""
- What are the side effects?
- What is the recommended dose?
- What are the contraindications?
- How should this drug be stored?
- What infections should I monitor for?
- What is the mechanism of action?
""")
    st.markdown("---")
    st.markdown("**🔧 Built with**")
    st.markdown("📄 PyPDF · 🔢 FAISS")
    st.markdown("🤗 sentence-transformers")
    st.markdown("🔗 LangChain · ⚡ Groq Llama 3")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("### 👈 Get started")
        st.markdown("Load the sample pharma PDF or upload your own using the sidebar, then ask any question.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.markdown("**📎 Document References:**")
                for i, src in enumerate(message["sources"]):
                    st.markdown(f'<span class="source-pill">Source {i+1}</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-text">{src[:250]}...</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                try:
                    response = get_answer(st.session_state.qa_chain, question)
                    answer = response["answer"]
                    sources = [doc.page_content for doc in response["sources"]]

                    st.markdown(answer)

                    st.markdown("**📎 Document References:**")
                    for i, src in enumerate(sources):
                        st.markdown(f'<span class="source-pill">Source {i+1}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="source-text">{src[:250]}...</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Something went wrong: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
