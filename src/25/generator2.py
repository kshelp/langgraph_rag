# ============================================================
# Generator Node
# 검색된 문서를 근거로 답변을 생성합니다.
# 재시도할 때는 이전 실패 이유를 프롬프트에 전달합니다.
# ============================================================

import os
import sys
import re
import warnings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# 현재 파일의 위치
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# src 폴더의 위치
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# src 폴더를 모듈 검색 경로에 추가
sys.path.insert(0, SRC_DIR)

# prompts.py가 있는 13차시 폴더 추가
sys.path.insert(0, os.path.join(SRC_DIR, "13"))

# retriever.py가 있는 24차시 폴더 추가
sys.path.insert(0, os.path.join(SRC_DIR, "24"))

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings("ignore")


# 프로젝트 설정 파일
import config

# 프롬프트 모음
from prompts import PROMPTS


# .env 파일 읽기
load_dotenv()


# 자료에서 답을 찾지 못했을 때 사용할 문구
NO_INFO = "자료에서 확인할 수 없습니다"


# 사용할 LLM 객체 생성
_llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
)


def create_basic_chain():
    """
    기본 답변 생성 체인을 만듭니다.

    config.PROMPT_VER에 지정된 프롬프트를 사용합니다.
    """

    # 설정 파일에서 사용할 프롬프트 버전을 가져옵니다.
    prompt_version = config.PROMPT_VER

    # 해당 버전의 프롬프트를 가져옵니다.
    prompt = PROMPTS[prompt_version]

    # 프롬프트와 LLM, 출력 변환기를 연결합니다.
    chain = prompt | _llm | StrOutputParser()

    return chain


def create_retry_chain():
    """
    재시도용 답변 생성 체인을 만듭니다.

    prompts.py에 등록된 retry 프롬프트를 사용합니다.
    """

    # 재시도 프롬프트를 가져옵니다.
    prompt = PROMPTS["retry"]

    # 재시도 프롬프트와 LLM을 연결합니다.
    chain = prompt | _llm | StrOutputParser()

    return chain


def build_context(documents):
    """
    검색된 문서들을 하나의 문자열로 합칩니다.

    각 문서에 [1], [2]와 같은 번호를 붙여서
    LLM이 답변에 출처를 표시할 수 있도록 합니다.
    """

    # 완성된 문서들을 저장할 목록입니다.
    context_parts = []

    # 검색된 문서를 하나씩 처리합니다.
    for index, document in enumerate(documents, start=1):

        # 문서의 파일명을 가져옵니다.
        filename = document.metadata.get("filename", "?")

        # 문서의 페이지 번호를 가져옵니다.
        page_no = document.metadata.get("page_no", "?")

        # 문서의 출처 정보를 만듭니다.
        source = f"[{index}] {filename} p.{page_no}"

        # 출처와 문서 내용을 합칩니다.
        document_text = source + "\n" + document.page_content

        # 완성된 문서를 목록에 추가합니다.
        context_parts.append(document_text)

    # 여러 문서를 구분선으로 연결합니다.
    context = "\n\n---\n\n".join(context_parts)

    return context


def check_citation(answer, document_count):
    """
    답변에 포함된 인용 번호가 올바른지 확인합니다.

    예:
        문서가 2개이고 [1], [2]를 사용하면 정상입니다.
        문서가 2개인데 [3]을 사용하면 잘못된 인용입니다.
    """

    # 답변에서 [숫자] 형태를 모두 찾습니다.
    found_numbers = re.findall(r"\[(\d+)\]", answer)

    # 인용 번호가 하나도 없으면 실패입니다.
    if not found_numbers:
        return False, "인용 없음"

    # 문자열로 찾은 번호를 정수로 변환합니다.
    citation_numbers = []

    for number in found_numbers:
        citation_numbers.append(int(number))

    # 잘못된 인용 번호를 저장할 목록입니다.
    invalid_numbers = []

    # 인용 번호가 실제 문서 범위 안에 있는지 확인합니다.
    for number in citation_numbers:

        # 문서 개수를 벗어난 번호인지 확인합니다.
        if number < 1 or number > document_count:
            invalid_numbers.append(number)

    # 잘못된 번호가 있으면 실패입니다.
    if invalid_numbers:
        message = f"존재하지 않는 근거 번호 {invalid_numbers}"
        return False, message

    # 모든 인용 번호가 정상인 경우입니다.
    message = f"인용 {len(citation_numbers)}건 정상"
    return True, message


def generator_node(state) -> dict:
    """
    검색된 문서를 바탕으로 답변을 생성합니다.

    첫 번째 실행에서는 기본 프롬프트를 사용합니다.
    재시도할 때는 실패 이유를 포함한 프롬프트를 사용합니다.
    """

    # State에서 검색된 문서를 가져옵니다.
    documents = state.get("documents") or []

    # 근거 문서가 없으면 LLM을 호출하지 않습니다.
    if not documents:
        return {
            "answer": config.MSG_NO_DOC,
            "insufficient": True,
            "has_citation": False,
            "log": [
                "생성: 근거 없음 - LLM 호출 생략"
            ],
        }

    # 현재 재시도 횟수를 가져옵니다.
    retries = state.get("retries", 0)

    # 이전 검증에서 기록한 실패 이유를 가져옵니다.
    reason = state.get("reason", "")

    # State에서 사용자 질문을 가져옵니다.
    question = state.get("question", "")

    # 검색 문서를 LLM 입력용 문자열로 변환합니다.
    context = build_context(documents)

    # 재시도 횟수와 실패 이유가 있으면 재시도 프롬프트를 사용합니다.
    if retries > 0 and reason:

        # 재시도 체인을 만듭니다.
        chain = create_retry_chain()

        # 재시도 프롬프트에 전달할 값을 준비합니다.
        payload = {
            "context": context,
            "question": question,
            "reason": reason,
        }

        # 로그에 표시할 프롬프트 종류입니다.
        prompt_type = "재시도 프롬프트"

    # 첫 번째 생성이면 기본 프롬프트를 사용합니다.
    else:

        # 기본 체인을 만듭니다.
        chain = create_basic_chain()

        # 기본 프롬프트에 전달할 값을 준비합니다.
        payload = {
            "context": context,
            "question": question,
        }

        # 로그에 표시할 프롬프트 종류입니다.
        prompt_type = "기본 프롬프트"

    try:
        # 프롬프트와 질문, 근거를 LLM에 전달합니다.
        answer = chain.invoke(payload)

    except Exception as error:
        # 오류의 종류를 문자열로 저장합니다.
        error_name = type(error).__name__

        # 생성 오류 결과를 반환합니다.
        return {
            "answer": config.MSG_ERROR,
            "insufficient": True,
            "has_citation": False,
            "gen_error": error_name,
            "log": [
                f"생성 오류: {error_name}"
            ],
        }

    # 답변에 포함된 인용 번호를 검사합니다.
    has_citation, citation_message = check_citation(
        answer,
        len(documents),
    )

    # 답변에 근거 부족 문구가 있는지 확인합니다.
    insufficient = NO_INFO in answer

    # 생성 결과를 반환합니다.
    return {
        "answer": answer,
        "insufficient": insufficient,
        "has_citation": has_citation,
        "log": [
            f"생성: {prompt_type}, "
            f"{len(answer)}자, "
            f"{citation_message}"
        ],
    }


# 이 파일을 직접 실행했을 때만 테스트합니다.
if __name__ == "__main__":

    # 24차시 Retriever Node를 가져옵니다.
    from retriever import retriever_node

    # 테스트할 질문입니다.
    question = "환불 방법과 수수료를 알려주세요"

    # 처음 사용할 State를 만듭니다.
    state = {
        "question": question,
        "query": question,
        "retries": 0,
        "reason": "",
    }

    # Retriever Node를 실행합니다.
    retrieved_data = retriever_node(state)

    # 검색 결과를 State에 추가합니다.
    state.update(retrieved_data)

    # 첫 번째 답변을 생성합니다.
    first_result = generator_node(state)

    # 첫 번째 결과를 출력합니다.
    print("\n[1회차 결과]")
    print(first_result["answer"])
    print(first_result["log"])

    # 첫 번째 생성 결과를 State에 추가합니다.
    state.update(first_result)

    # Verifier가 기록했다고 가정하는 값입니다.
    state["retries"] = 1
    state["reason"] = "답변을 1문장으로 줄이세요"

    # 실패 이유를 반영하여 다시 생성합니다.
    second_result = generator_node(state)

    # 두 번째 결과를 출력합니다.
    print("\n[2회차 결과]")
    print(second_result["answer"])
    print(second_result["log"])