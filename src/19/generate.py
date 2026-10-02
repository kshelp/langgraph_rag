# ============================================================
# Generator Node
# 검색된 문서를 근거로 LLM에게 답변을 생성하도록 요청합니다.
# 답변과 함께 자료 부족 여부와 인용 번호 정상 여부를 반환합니다.
# ============================================================

import os
import sys
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import config


# ------------------------------------------------------------
# 13차시에 만든 prompts.py를 사용하기 위해 경로를 추가합니다.
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(CURRENT_DIR, "..", "13")

sys.path.insert(0, PROMPT_DIR)

from prompts import PROMPTS


# ------------------------------------------------------------
# .env 파일의 환경 변수를 불러옵니다.
# ------------------------------------------------------------
load_dotenv()


# ------------------------------------------------------------
# 답변 생성에 사용할 LLM 체인을 만듭니다.
# 프롬프트 → LLM → 문자열 변환 순서로 실행됩니다.
# ------------------------------------------------------------
prompt = PROMPTS[config.PROMPT_VER]

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE
)

chain = prompt | llm | StrOutputParser()


# ------------------------------------------------------------
# 자료에 답이 없을 때 사용하는 문구입니다.
# ------------------------------------------------------------
NO_INFO = "자료에서 확인할 수 없습니다"


# ============================================================
# 검색 문서를 LLM에게 전달할 문자열로 변환
# ============================================================
def build_context(documents):
    """검색된 문서들을 하나의 문자열로 합칩니다."""

    parts = []

    # 문서를 하나씩 꺼냅니다.
    for index in range(len(documents)):

        document = documents[index]

        # 문서의 metadata를 가져옵니다.
        metadata = document.metadata

        if not metadata:
            metadata = {}

        # 파일명을 찾습니다.
        filename = metadata.get("filename")

        if not filename:
            filename = metadata.get("file_name")

        if not filename:
            filename = metadata.get("source")

        if not filename:
            filename = "unknown"

        # 페이지 번호를 찾습니다.
        page_no = metadata.get("page_no")

        if page_no is None:
            page_no = metadata.get("page")

        if page_no is None:
            page_no = metadata.get("pageno")

        if page_no is None:
            page_no = "?"

        # 문서 번호는 1번부터 시작합니다.
        document_number = index + 1

        # LLM에게 전달할 문서 내용을 만듭니다.
        text = (
            f"[{document_number}] {filename} p.{page_no}\n"
            f"{document.page_content}"
        )

        parts.append(text)

    # 여러 문서를 구분선으로 연결합니다.
    context = "\n\n---\n\n".join(parts)

    return context


# ============================================================
# Generator Node
# ============================================================
def generate_node(state: dict) -> dict:
    """검색된 문서를 근거로 답변을 생성합니다."""

    # Retriever Node가 저장한 문서를 가져옵니다.
    documents = state.get("documents")

    if not documents:
        documents = []

    # 검색 문서가 없으면 LLM을 호출하지 않습니다.
    if len(documents) == 0:
        return {
            "answer": config.MSG_NO_DOC,
            "insufficient": True,
            "has_citation": False,
            "log": ["[generate] 근거 없음 - 생성 생략"],
        }

    # 검색 문서를 LLM에게 전달할 문자열로 만듭니다.
    context = build_context(documents)

    # 사용자의 원래 질문을 가져옵니다.
    question = state["question"]

    # 문서와 질문을 LLM에게 전달하여 답변을 생성합니다.
    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    # 답변에서 [1], [2]와 같은 인용 번호를 찾습니다.
    found_numbers = re.findall(r"\[(\d+)\]", answer)

    # 찾은 인용 번호를 숫자로 변환합니다.
    citation_numbers = []

    for number in found_numbers:
        citation_numbers.append(int(number))

    # 인용 번호가 정상인지 검사합니다.
    has_citation = False

    if len(citation_numbers) > 0:
        has_citation = True

        # 존재하지 않는 문서 번호가 있으면 비정상입니다.
        for number in citation_numbers:
            if number < 1 or number > len(documents):
                has_citation = False
                break

    # 자료 부족 문구가 답변에 있는지 확인합니다.
    insufficient = NO_INFO in answer

    # 로그에 표시할 인용 상태를 정합니다.
    if has_citation:
        citation_status = "정상"
    else:
        citation_status = "미확인"

    # 생성 결과를 State에 반영할 수 있도록 반환합니다.
    return {
        "answer": answer,
        "insufficient": insufficient,
        "has_citation": has_citation,
        "log": [
            f"[generate] {len(answer)}자 생성 "
            f"(인용 {citation_status})"
        ],
    }


# ============================================================
# Generator Node 단독 테스트
# ============================================================
if __name__ == "__main__":

    # 앞에서 만든 Retriever Node를 가져옵니다.
    from retrieve import retrieve_node

    # 테스트용 State를 만듭니다.
    state = {
        "question": "환불은 며칠 이내인가요?"
    }

    # 먼저 Retriever Node를 실행합니다.
    retrieve_result = retrieve_node(state)

    # 검색 결과를 기존 State에 추가합니다.
    state.update(retrieve_result)

    # 검색 결과를 이용하여 Generator Node를 실행합니다.
    generate_result = generate_node(state)

    # 생성된 답변을 출력합니다.
    print(generate_result["answer"])