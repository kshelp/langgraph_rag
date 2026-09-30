from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, sys

CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")
sys.path.insert(0, INGEST_DIR)

from ingest import load_documents

docs = load_documents("../../data/manual.pdf")

spitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function = len,
)

chunks = spitter.split_documents(docs)

print(len(chunks))
print((chunks[0].page_content))
print((chunks[0].metadata))


