import os
import re
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

CURRENT_DIR = os.path.dirname(__file__)
RETRIEVER_DIR = os.path.join(CURRENT_DIR, "..", "11")

sys.path.append(RETRIEVER_DIR)

from retriever import search, build_context
from prompts import RAG_PROMPT_V3
from validators import check_citation

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

_chain = RAG_PROMPT_V3 | llm | StrOutputParser()

NO_INFO = "자료에서 확인할 수 없습니다"

def ask(question, k=3, verbose=True):
    documents = search(question, k=k)

    if verbose:
        print("Q:", question)

    if not documents:
        result = {
            "answer": "관련 자료를 찾지 못했습니다.",
            "sources": [],
            "cited": False,
            "insufficient": True
        }

        if verbose:
            print("A:", result["answer"])
            print("⚠️ 검색 결과 없음")
            print("-" * 55)

        return result

    context = build_context(documents)

    answer = _chain.invoke({
        "context": context,
        "question": question
    })

    cited, citation_message = check_citation(
        answer,
        len(documents)
    )

    sources = []

    for document in documents:
        sources.append(document.metadata)

    return {
        "answer": answer,
        "sources": sources,
        "cited": cited,
        "insufficient": NO_INFO in answer
    }

if __name__ == "__main__":
    ask("환불은 며칠 이내에 신청해야 하나요?")
    ask("대표이사가 누구인가요?")
    ask("환불 방법과 수수료를 알려주세요")