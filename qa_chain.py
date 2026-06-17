from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from prompts import QA_PROMPT
import os

def get_llm():
    load_dotenv()
    groq_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(api_key=groq_key,
    model="llama-3.1-8b-instant",
    temperature=0)
    return llm

def create_qa_chain(retriever):
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type (llm=llm, 
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True)
    return qa_chain

def get_answer(qa_chain, question: str) -> dict:
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }

if __name__ == '__main__':
    from pdf_processor import process_pdf
    from embeddings import create_embeddings, get_retriever

    chunks = process_pdf("sample_docs/sample.pdf")
    vector_store = create_embeddings(chunks)
    retriever = get_retriever(vector_store)
    qa_chain = create_qa_chain(retriever)

    question = "What are the side effects of GAMIFANT?"
    response = get_answer(qa_chain, question)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources used: {len(response['sources'])} chunks")




