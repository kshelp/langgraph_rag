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

store = FAISS.load_local(
    INDEX_PATH,
    emb,
    allow_dangerous_deserialization=True
)

print("인덱스 로드 완료")

query = "환불 규정"

found = store.similarity_search(
    query,
    k=3
)

print(f"검색 결과 : {len(found)}건")

for i, doc in enumerate(found, 1):
    print(f"\n[{i}]")
    print(doc.page_content)