# ============================================================
# Verifier Node
# ============================================================
# Generator가 만든 답변을 근거 자료와 비교하여 검사합니다.
# 문제가 있으면 재생성(retry) 또는 재검색(research)을 요청합니다.
# ============================================================


import os
import sys
import json
import re


# ------------------------------------------------------------
# 현재 파일의 위치를 기준으로 필요한 폴더를 찾습니다.
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# src 폴더 경로입니다.
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# 25차시 Generator 폴더 경로입니다.
GENERATOR_DIR = os.path.join(CURRENT_DIR, "..", "25")

# JUDGE_PROMPT가 있는 13차시 경로
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "13"))

# 24차시 Retriever 폴더 경로입니다.
RETRIEVER_DIR = os.path.join(CURRENT_DIR, "..", "24")

# src 폴더를 파이썬 모듈 검색 경로에 추가합니다.
sys.path.insert(0, SRC_DIR)

# Generator 폴더를 검색 경로에 추가합니다.
sys.path.insert(0, GENERATOR_DIR)

# Retriever 폴더를 검색 경로에 추가합니다.
sys.path.insert(0, RETRIEVER_DIR)


# ------------------------------------------------------------
# Verifier에서 사용할 모듈을 가져옵니다.
# ------------------------------------------------------------

# 심판용 프롬프트를 가져옵니다.
from prompts import JUDGE_PROMPT

# Generator에서 만든 문서 조합 함수와 LLM을 가져옵니다.
from generator import build_context, _llm

# LLM 응답을 문자열로 변환하는 파서입니다.
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------------------
# 심판용 LLM 체인을 만듭니다.
# ------------------------------------------------------------

# 프롬프트 → LLM 호출 → 문자열 변환 순서로 실행됩니다.
judge_chain = JUDGE_PROMPT | _llm | StrOutputParser()


# ============================================================
# LLM 응답에서 JSON을 추출하는 함수
# ============================================================

def parse_json(raw_text: str) -> dict:
    """
    LLM 응답에서 JSON 부분만 찾아 딕셔너리로 변환합니다.

    LLM이 JSON 앞뒤에 설명을 덧붙여도
    중괄호로 둘러싸인 부분만 추출합니다.
    """

    # 응답 안에서 { }로 둘러싸인 부분을 찾습니다.
    match = re.search(r"\{.*\}", raw_text, re.S)

    # JSON 형식을 찾지 못하면 오류를 발생시킵니다.
    if match is None:
        raise ValueError("JSON 블록을 찾을 수 없습니다.")

    # 찾은 JSON 문자열을 파이썬 딕셔너리로 변환합니다.
    return json.loads(match.group())


# ============================================================
# Verifier Node
# ============================================================

def verifier_node(state: dict) -> dict:
    """
    Generator가 만든 답변을 검증합니다.

    1단계에서는 규칙 기반 검사를 수행합니다.
    2단계에서는 LLM 심판에게 답변 평가를 요청합니다.
    """

    # 검색된 문서를 가져옵니다.
    documents = state.get("documents") or []

    # Generator가 만든 답변을 가져옵니다.
    answer = state.get("answer", "")

    # 사용자 질문을 가져옵니다.
    question = state.get("question", "")


    # --------------------------------------------------------
    # 1차 검사: 비용이 발생하지 않는 규칙 기반 검사
    # --------------------------------------------------------

    # 답변 생성 중 오류가 발생했으면 중단합니다.
    if state.get("gen_error"):
        return {
            "grade": "giveup",
            "reason": "답변 생성 중 오류가 발생했습니다.",
            "log": ["검증: 생성 오류로 중단"]
        }

    # Generator가 근거 부족을 신고했으면 다시 검색합니다.
    if state.get("insufficient"):
        return {
            "grade": "research",
            "reason": "Generator가 근거 부족을 신고했습니다.",
            "log": ["검증(규칙): 근거 부족"]
        }

    # 답변에 인용 번호가 없으면 다시 생성합니다.
    if not state.get("has_citation"):
        return {
            "grade": "retry",
            "reason": "출처 번호가 없거나 유효하지 않습니다.",
            "log": ["검증(규칙): 인용 불량"]
        }

    # 답변이 너무 짧으면 다시 생성합니다.
    if len(answer.strip()) < 10:
        return {
            "grade": "retry",
            "reason": "답변이 너무 짧습니다.",
            "log": ["검증(규칙): 답변 길이 부족"]
        }


    # --------------------------------------------------------
    # 2차 검사: LLM 심판을 이용한 의미 기반 검사
    # --------------------------------------------------------

    try:
        # 검색 문서를 하나의 문자열로 합칩니다.
        context = build_context(documents)

        # 심판 LLM에 근거, 질문, 답변을 전달합니다.
        raw_result = judge_chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })

        # LLM 응답에서 JSON 판정 결과를 추출합니다.
        judge_result = parse_json(raw_result)

    except Exception as error:
        # 검증 오류 때문에 전체 서비스가 중단되지 않도록 통과시킵니다.
        error_name = type(error).__name__

        return {
            "grade": "pass",
            "reason": f"검증 불가({error_name}) - 통과 처리",
            "log": [f"검증 오류: {error_name}"]
        }


    # --------------------------------------------------------
    # LLM 심판 결과를 읽습니다.
    # --------------------------------------------------------

    # 답변이 근거 자료에 포함되어 있는지 확인합니다.
    grounded = bool(judge_result.get("grounded"))

    # 답변이 질문에 적절히 답했는지 확인합니다.
    relevant = bool(judge_result.get("relevant"))

    # 심판 이유를 최대 200자까지만 저장합니다.
    reason = str(judge_result.get("reason", ""))[:200]


    # --------------------------------------------------------
    # 검증 결과에 따라 다음 작업을 결정합니다.
    # --------------------------------------------------------

    # 근거에도 맞고 질문에도 맞으면 통과시킵니다.
    if grounded and relevant:
        grade = "pass"

    # 근거에 없는 내용이 있으면 답변을 다시 생성합니다.
    elif not grounded:
        grade = "retry"

    # 근거는 있지만 질문에 맞지 않으면 다시 검색합니다.
    else:
        grade = "research"


    # --------------------------------------------------------
    # 검증 결과를 State에 기록합니다.
    # --------------------------------------------------------

    return {
        "grade": grade,
        "reason": reason,
        "log": [
            f"검증(LLM): grounded={grounded}, "
            f"relevant={relevant} → {grade}"
        ]
    }


# ============================================================
# 단독 테스트
# ============================================================

if __name__ == "__main__":

    # Retriever Node를 가져옵니다.
    from retriever import retriever_node


    # 테스트에 사용할 질문입니다.
    question = "환불은 며칠 이내인가요?"

    # 검색에 필요한 초기 State입니다.
    state = {
        "question": question,
        "query": question
    }

    # 질문에 대한 근거 문서를 검색합니다.
    retrieval_result = retriever_node(state)

    # 검색 결과를 기존 State에 합칩니다.
    state.update(retrieval_result)


    # Verifier가 제대로 오류를 잡는지 확인하기 위한 테스트입니다.
    test_cases = [
        {
            "name": "정상 답변",
            "answer": "환불은 상품 수령 후 7일 이내에 신청하실 수 있습니다[1].",
            "insufficient": False,
            "has_citation": True
        },
        {
            "name": "환각 포함",
            "answer": "환불은 7일 이내입니다[1]. 수수료는 상품가의 10%입니다[1].",
            "insufficient": False,
            "has_citation": True
        },
        {
            "name": "동문서답",
            "answer": "고객센터는 평일 09시부터 18시까지 운영합니다[1].",
            "insufficient": False,
            "has_citation": True
        },
        {
            "name": "인용 없음",
            "answer": "환불은 7일 이내에 신청하시면 됩니다.",
            "insufficient": False,
            "has_citation": False
        },
        {
            "name": "근거 부족 신고",
            "answer": "자료에서 확인할 수 없습니다.",
            "insufficient": True,
            "has_citation": False
        }
    ]


    # 각각의 테스트 케이스를 실행합니다.
    for test_case in test_cases:

        # 공통 State를 복사합니다.
        test_state = dict(state)

        # 테스트할 답변 정보를 추가합니다.
        test_state["answer"] = test_case["answer"]
        test_state["insufficient"] = test_case["insufficient"]
        test_state["has_citation"] = test_case["has_citation"]

        # Verifier Node를 실행합니다.
        result = verifier_node(test_state)

        # 테스트 결과를 출력합니다.
        print()
        print(f"[{test_case['name']}]")
        print(f"  판정: {result['grade']}")
        print(f"  이유: {result['reason'][:70]}")