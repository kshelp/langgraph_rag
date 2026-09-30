from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PREPARE_DIR = os.path.join(CURRENT_DIR, "..", "07")
sys.path.insert(0, PREPARE_DIR)
from prepare import prepare_chunks

load_dotenv()

DOC_PATH = "../../data/manual.pdf"
INDEX_PATH = "faiss_index"
EMBED_MODEL = "text-embedding-3-small"

chunks = prepare_chunks(DOC_PATH)
print(f"문서 조각 수 : {len(chunks)}개")

emb = OpenAIEmbeddings(model="text-embedding-3-small")

print("임베딩 중...")

store = FAISS.from_documents(chunks, emb)

store.save_local(INDEX_PATH)

print("인덱스를 저장했습니다.")