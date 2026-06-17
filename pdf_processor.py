from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

def process_pdf(pdf_path: str) -> list:
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    return chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        chunks = process_pdf(sys.argv[1])
        print(f"Total chunks created: {len(chunks)}")
        print(f"\nFirst chunk preview:")
        print(chunks[0][:300])
    else:
        print("Usage: python pdf_processor.py path/to/file.pdf")
