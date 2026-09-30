from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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

full_text = ""

for doc in docs:
    full_text += doc.page_content + "\n\n"

merged_doc = Document(page_content=full_text)

chunks = spitter.split_documents([merged_doc])

def show_boundary(chunks, index=0):
    print(chunks[index].page_content[-80:])
    print()
    print(chunks[index+1].page_content[:80])

show_boundary(chunks, 0)

