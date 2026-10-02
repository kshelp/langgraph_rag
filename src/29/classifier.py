# ============================================================
# Classifier Node
# ============================================================
# 사용자의 질문을 인사, 계산, 범위 밖 질문, 문서 질문으로 분류합니다.
# 먼저 규칙으로 판단하고, 판단하기 어려운 질문만 LLM에게 전달합니다.
# ============================================================


import os
import sys
import re


# ============================================================
# 다른 차시의 파일을 import하기 위한 경로 설정
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# 공통 src 폴더를 추가합니다.
sys.path.insert(0, SRC_DIR)

# 25차시 Generator 모듈 경로를 추가합니다.
sys.path.insert(0, os.path.join(SRC_DIR, "25"))

# 13차시 프롬프트 모듈 경로를 추가합니다.
sys.path.insert(0, os.path.join(SRC_DIR, "13"))

# 24차시 Retriever 모듈 경로를 추가합니다.
sys.path.insert(0, os.path.join(SRC_DIR, "24"))


# 다른 모듈을 가져옵니다.
import config

from generator import _llm
from prompts import CLASSIFY_PROMPT
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 분류 기준
# ============================================================

# 인사말로 판단할 단어 목록입니다.
GREETING_WORDS = [
    "안녕",
    "반가",
    "하이",
    "헬로",
    "고마",
    "감사",
    "수고",
    "잘 부탁",
]

# 문서와 관련이 없다고 판단할 단어 목록입니다.
OUT_OF_SCOPE_WORDS = [
    "날씨",
    "주가",
    "환율",
    "로또",
    "뉴스",
    "맛집",
    "영화",
    "축구",
]

# 숫자와 계산 기호로만 이루어진 질문인지 확인합니다.
CALC_PATTERN = re.compile(
    r"^[\d\s+\-*/().,]+[=?]?$"
)


# ============================================================
# 1차 분류: 규칙 기반 분류
# ============================================================

def rule_classify(question):
    """
    규칙만 사용하여 질문을 빠르게 분류합니다.

    반환값:
        greeting : 인사말
        calc     : 계산식
        scope    : 문서 범위 밖 질문
        None     : 규칙으로 판단하지 못함
    """

    # 질문 앞뒤의 공백을 제거합니다.
    question = question.strip()

    # 빈 질문은 문서와 관련 없는 질문으로 처리합니다.
    if question == "":
        return "scope"

    # --------------------------------------------------------
    # 1. 인사말인지 확인합니다.
    # --------------------------------------------------------

    # 긴 문장 전체를 인사말로 잘못 판단하지 않도록 제한합니다.
    if len(question) <= 20:

        # 인사말 단어를 하나씩 확인합니다.
        for word in GREETING_WORDS:

            # 질문에 인사말 단어가 포함되어 있으면 인사로 분류합니다.
            if word in question:
                return "greeting"

    # --------------------------------------------------------
    # 2. 계산식인지 확인합니다.
    # --------------------------------------------------------

    # 계산 연산자가 있는지 저장하는 변수입니다.
    has_operator = False

    # 사칙연산 기호를 하나씩 확인합니다.
    for operator in "+-*/":

        # 질문에 연산자가 있으면 계산식 후보로 표시합니다.
        if operator in question:
            has_operator = True
            break

    # 계산식 모양이고 실제 연산자도 있으면 계산으로 분류합니다.
    if CALC_PATTERN.match(question) and has_operator:
        return "calc"

    # --------------------------------------------------------
    # 3. 문서 범위 밖 질문인지 확인합니다.
    # --------------------------------------------------------

    # 범위 밖 단어를 하나씩 확인합니다.
    for word in OUT_OF_SCOPE_WORDS:

        # 질문에 범위 밖 단어가 포함되어 있으면 scope로 분류합니다.
        if word in question:
            return "scope"

    # 규칙으로 판단하지 못한 질문입니다.
    return None


# ============================================================
# 2차 분류: LLM 기반 분류
# ============================================================

# 프롬프트, LLM, 문자열 변환기를 연결합니다.
classify_chain = (
    CLASSIFY_PROMPT
    | _llm
    | StrOutputParser()
)


# LLM이 반환할 수 있는 정상적인 분류 결과입니다.
VALID_INTENTS = [
    "greeting",
    "calc",
    "scope",
    "document",
]


# ============================================================
# LangGraph 분류 노드
# ============================================================

def classifier_node(state):
    """
    질문을 분류하고 intent 값을 State에 저장합니다.

    처리 순서:
        1. 규칙으로 먼저 판단합니다.
        2. 판단하지 못하면 LLM 사용 여부를 확인합니다.
        3. LLM을 사용하지 않으면 document로 처리합니다.
        4. LLM 오류가 발생해도 document로 처리합니다.
    """

    # State에서 질문을 가져옵니다.
    question = state.get("question", "")

    # --------------------------------------------------------
    # 1단계: 규칙 기반 분류
    # --------------------------------------------------------

    # 비용이 들지 않는 규칙 분류를 먼저 실행합니다.
    intent = rule_classify(question)

    # 규칙으로 판단했다면 즉시 결과를 반환합니다.
    if intent is not None:
        return {
            "intent": intent,
            "log": [f"분류(규칙): {intent}"],
        }

    # --------------------------------------------------------
    # 2단계: LLM 분류 사용 여부 확인
    # --------------------------------------------------------

    # 설정값이 없으면 기본적으로 LLM을 사용하지 않습니다.
    use_llm = getattr(config, "USE_LLM_CLASSIFY", False)

    # LLM 분류를 사용하지 않으면 문서 질문으로 처리합니다.
    if use_llm == False:
        return {
            "intent": "document",
            "log": ["분류: 기본값 document"],
        }

    # --------------------------------------------------------
    # 3단계: LLM으로 분류
    # --------------------------------------------------------

    try:
        # LLM에게 질문을 전달합니다.
        result = classify_chain.invoke({
            "question": question,
        })

        # LLM의 결과를 소문자로 바꾸고 공백을 제거합니다.
        intent = result.strip().lower()

        # 허용되지 않은 결과는 document로 바꿉니다.
        if intent not in VALID_INTENTS:
            intent = "document"

        # LLM 분류 결과를 기록합니다.
        log_message = f"분류(LLM): {intent}"

    except Exception as error:
        # LLM 오류가 발생하면 안전하게 document로 처리합니다.
        intent = "document"

        # 오류 종류만 로그에 기록합니다.
        error_name = type(error).__name__
        log_message = f"분류 오류({error_name}) -> document"

    # 최종 분류 결과를 State에 저장합니다.
    return {
        "intent": intent,
        "log": [log_message],
    }


# ============================================================
# 단독 실행 테스트
# ============================================================

if __name__ == "__main__":

    # 테스트할 질문 목록입니다.
    test_questions = [
        "안녕하세요",
        "125 + 340 * 2",
        "오늘 날씨 어때요?",
        "환불은 며칠 이내인가요?",
    ]

    # 질문을 하나씩 테스트합니다.
    for question in test_questions:

        # 테스트용 State를 직접 만듭니다.
        state = {
            "question": question
        }

        # 분류 노드를 실행합니다.
        result = classifier_node(state)

        # 결과를 출력합니다.
        print(f"\n질문: {question}")
        print(f"분류: {result['intent']}")
        print(f"로그: {result['log']}")