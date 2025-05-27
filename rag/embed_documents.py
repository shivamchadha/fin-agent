import os
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup

# Define the folder containing your PDFs and EPUBs
DATA_DIR = "./data/finance_resources"

def extract_text_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return text

def extract_text_epub(file_path):
    text = ""
    try:
        book = epub.read_epub(file_path)
        for item in book.get_items():
            if isinstance(item, epub.EpubHtml):
                soup = BeautifulSoup(item.get_content(), features="html.parser")
                text += soup.get_text(separator=" ", strip=True) + " "
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return text

def compute_hash(text):
    """Compute an MD5 hash of the text for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def main():
    documents = []  
    metadata = []   
    seen_hashes = set()

    # Iterate over files in the data directory
    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            ext = filename.lower().split('.')[-1]
            file_path = os.path.join(root, filename)
            text = ""
            if ext == "pdf":
                text = extract_text_pdf(file_path)
            elif ext == "epub":
                text = extract_text_epub(file_path)
            else:
                continue

            # Skip files with no extracted text
            if not text.strip():
                continue

            # Compute a hash to detect duplicates
            text_hash = compute_hash(text)
            if text_hash in seen_hashes:
                print(f"Duplicate found: {file_path}")
                continue

            seen_hashes.add(text_hash)
            documents.append(text)
            metadata.append({
                "file_path": file_path,
                "hash": text_hash,
                "length": len(text)
            })

    if not documents:
        print("No documents to process.")
        exit(0)

    # Initialize SentenceTransformer embeddings using the new import path.
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build the FAISS vector store using the helper method. This method
    # creates the index, stores document texts, and associates metadata.
    vector_db = FAISS.from_texts(texts=documents, embedding=embeddings, metadatas=metadata)
    print(f"Added {len(documents)} documents to the vector store.")

    # Save the vector store locally. This creates a directory (e.g. 'vector_db.index')
    # with all the required files (e.g., index.faiss, index.pkl, and metadata.json).
    vector_db.save_local("vector_db.index")
    print("Vector DB saved locally in the 'vector_db.index' directory.")

if __name__ == "__main__":
    main()
