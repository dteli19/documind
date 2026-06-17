from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_embeddings(chunks: list) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_vector_store = FAISS.from_texts(chunks, embeddings)
    return faiss_vector_store

def get_retriever(faiss_vector_store: FAISS):
    retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever

if __name__ == '__main__':
    from pdf_processor import process_pdf
    chunks = process_pdf("sample_docs/sample.pdf")
    print(f"Creating embeddings for {len(chunks)} chunks...")
    vector_store = create_embeddings(chunks)
    print("Vector store created successfully!")
    retriever = get_retriever(vector_store)
    results = retriever.invoke("What are the side effects of GAMIFANT?")
    print(f"\nTop {len(results)} relevant chunks found:")
    for i, doc in enumerate(results):
        print(f"\nChunk {i+1}:\n{doc.page_content[:200]}")





