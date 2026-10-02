# ============================================================
# Generator Node
# 검색된 문서를 바탕으로 답변을 생성합니다.
# 근거가 없으면 LLM을 호출하지 않고 안내 메시지를 반환합니다.
# ============================================================

import os
import sys
import re
import warnings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 현재 파일의 폴더 위치
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# src 폴더 위치
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


# LLM이 근거를 찾지 못했을 때 사용할 문구
NO_INFO = "자료에서 확인할 수 없습니다"


# 사용할 LLM 객체 생성
_llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
)


def create_chain():
    """
    설정 파일에 지정된 프롬프트와 LLM을 연결합니다.

    반환값:
        프롬프트 → LLM → 문자열 변환기가 연결된 실행 체인
    """

    # config.py에서 사용할 프롬프트 버전 가져오기
    prompt_version = config.PROMPT_VER

    # 선택한 프롬프트 가져오기
    prompt = PROMPTS[prompt_version]

    # 프롬프트, LLM, 출력 변환기를 연결
    chain = prompt | _llm | StrOutputParser()

    return chain


def build_context(documents):
    """
    검색된 문서들을 LLM에게 전달할 하나의 문자열로 합칩니다.

    각 문서 앞에 [1], [2]와 같은 번호와
    파일명, 페이지 번호를 붙입니다.
    """

    context_parts = []

    # 검색된 문서를 하나씩 처리
    for index, document in enumerate(documents, start=1):

        # 파일명을 가져옵니다.
        filename = document.metadata.get("filename", "?")

        # 페이지 번호를 가져옵니다.
        page_no = document.metadata.get("page_no", "?")

        # 문서의 출처 정보를 만듭니다.
        source = f"[{index}] {filename} p.{page_no}"

        # 출처와 문서 내용을 하나로 합칩니다.
        text = source + "\n" + document.page_content

        # 완성된 문서를 목록에 추가합니다.
        context_parts.append(text)

    # 여러 문서를 구분선으로 연결합니다.
    context = "\n\n---\n\n".join(context_parts)

    return context


def check_citation(answer, document_count):
    """
    답변에 포함된 인용 번호가 정상인지 확인합니다.

    예:
        [1], [2] → 정상
        인용 없음 → 실패
        문서가 2개인데 [3] → 실패

    반환값:
        (검사 결과, 검사 메시지)
    """

    # 답변에서 [숫자] 형태의 인용 번호를 찾습니다.
    numbers = re.findall(r"\[(\d+)\]", answer)

    # 인용 번호가 하나도 없으면 실패입니다.
    if not numbers:
        return False, "인용 없음"

    # 문자열로 찾은 번호를 정수로 변환합니다.
    citation_numbers = []

    for number in numbers:
        citation_numbers.append(int(number))

    # 존재하지 않는 인용 번호를 저장할 목록입니다.
    invalid_numbers = []

    # 인용 번호가 실제 문서 개수 안에 있는지 확인합니다.
    for number in citation_numbers:

        # 문서가 2개인데 [3]을 사용하면 잘못된 번호입니다.
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

    입력:
        state["question"]  : 사용자 질문
        state["documents"] : 검색된 문서 목록

    출력:
        answer             : 생성된 답변
        insufficient       : 근거 부족 여부
        has_citation       : 인용 번호 정상 여부
        log                : 처리 결과 기록
    """

    # State에서 검색 문서 목록을 가져옵니다.
    documents = state.get("documents") or []

    # 검색된 문서가 없으면 LLM을 호출하지 않습니다.
    if not documents:
        return {
            "answer": config.MSG_NO_DOC,
            "insufficient": True,
            "has_citation": False,
            "log": [
                "생성: 근거 없음 - LLM 호출 생략"
            ],
        }

    # State에서 사용자 질문을 가져옵니다.
    question = state.get("question", "")

    # 검색된 문서를 LLM 입력용 문자열로 변환합니다.
    context = build_context(documents)

    try:
        # 프롬프트와 LLM을 연결한 체인을 만듭니다.
        chain = create_chain()

        # 질문과 근거를 LLM에 전달하여 답변을 생성합니다.
        answer = chain.invoke({
            "context": context,
            "question": question,
        })

    # LLM 호출 중 오류가 발생하면 안내 결과를 반환합니다.
    except Exception as error:
        error_name = type(error).__name__

        return {
            "answer": config.MSG_ERROR,
            "insufficient": True,
            "has_citation": False,
            "gen_error": error_name,
            "log": [
                f"생성 오류: {error_name}"
            ],
        }

    # 답변의 인용 번호를 검사합니다.
    has_citation, citation_message = check_citation(
        answer,
        len(documents),
    )

    # 답변에 근거 부족 문구가 포함되어 있는지 확인합니다.
    insufficient = NO_INFO in answer

    # Generator Node의 결과를 반환합니다.
    return {
        "answer": answer,
        "insufficient": insufficient,
        "has_citation": has_citation,
        "log": [
            f"생성: {len(answer)}자, {citation_message}"
        ],
    }


# 이 파일을 직접 실행했을 때만 테스트합니다.
if __name__ == "__main__":

    # 24차시 Retriever Node를 가져옵니다.
    from retriever import retriever_node

    # 테스트할 질문 목록입니다.
    questions = [
        "환불은 며칠 이내에 신청해야 하나요?",
        "환불 방법과 수수료를 알려주세요",
        "대표이사가 누구인가요?",
    ]

    # 질문을 하나씩 테스트합니다.
    for question in questions:

        # 처음에는 질문만 State에 넣습니다.
        state = {
            "question": question,
            "query": question,
        }

        # Retriever Node를 실행하고 검색 결과를 State에 추가합니다.
        retrieved_data = retriever_node(state)
        state.update(retrieved_data)

        # Generator Node를 실행합니다.
        result = generator_node(state)

        # 결과를 출력합니다.
        print("\n" + "=" * 55)
        print(f"질문: {question}")
        print(f"근거 문서: {len(state.get('documents', []))}건")
        print(f"답변: {result['answer'][:110]}")
        print(f"근거 부족: {result['insufficient']}")
        print(f"인용 정상: {result['has_citation']}")
        print(f"로그: {result['log'][0]}")