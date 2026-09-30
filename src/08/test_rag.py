from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "07")
sys.path.insert(0, INGEST_DIR)

from prepare import prepare_chunks

load_dotenv()

chunks = prepare_chunks("../../data/manual.pdf")

print(f"문서 분할 완료 : {len(chunks)}개")

emb = OpenAIEmbeddings(model="text-embedding-3-small")

print("임베딩 중...(조금 걸립니다)")

store = FAISS.from_documents(chunks, emb)

print("인덱싱 완료")

# 답변을 생성할 LLM 준비
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("준비 완료\n")

def build_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        text = (
            f"[{i}] "
            f"({doc.metadata['filename']} "
            f"p.{doc.metadata['page_no']})\n"
            f"{doc.page_content}"
        )
        parts.append(text)
    context = "\n\n".join(parts)
    return context

def ask(question, k=3):
    found = store.similarity_search(question, k=k)
    if not found:
        print("관련 자료를 찾지 못했습니다.")
        return
    context = build_context(found)

    prompt = (
        "아래 자료만 근거로 답하세요.\n"
        "자료에 없는 내용은 '자료에서 확인할 수 없습니다'라고 답하세요.\n"
        "추측하지 마세요.\n\n"
        f"[자료]\n{context}\n\n"
        f"[질문] {question}"
    )

    response = llm.invoke(prompt)
    answer = response.content

    for doc in found:
        print(
            f"{doc.metadata['filename']} "
            f"{doc.metadata['page_no']}"
        )

    return answer, found


TESTS = [
    "환불은 며칠 이내에 신청해야 하나요?",
    "우리 회사 대표이사 이름이 뭔가요?",
    "반품 절차를 알려주세요",
    "환불과 교환은 어떻게 다른가요?",
]

for i, question in enumerate(TESTS, 1):
    print(f"질문 {i}: {question}")
    answer, found = ask(question)
    print(f"답변 {i}: {answer}")
    print()

