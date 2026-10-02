# ============================================================
# Manual Flow
# LangGraph를 사용하지 않고 Node들을 직접 순서대로 실행합니다.
# 검색 성공 여부에 따라 Generator 또는 Fallback으로 분기합니다.
# ============================================================

import warnings

from retrieve import retrieve_node
from generate import generate_node
from fallback import fallback_node


# 불필요한 경고 메시지를 숨깁니다.
warnings.filterwarnings("ignore")


# ============================================================
# Node들을 직접 연결하여 실행
# ============================================================
def manual_run(question: str):
    """그래프 없이 Node들을 직접 연결하여 실행합니다."""

    # 처음 사용할 State를 만듭니다.
    state = {
        "question": question,
        "query": question,
        "retries": 0,
        "log": [],
        "tried_queries": [],
    }

    # --------------------------------------------------------
    # 1. Retriever Node 실행
    # --------------------------------------------------------

    # 질문과 관련된 문서를 검색합니다.
    result = retrieve_node(state)

    # 검색 과정의 로그를 따로 보관합니다.
    new_log = state["log"] + result["log"]

    # 검색 결과를 State에 추가합니다.
    state.update(result)

    # 기존 로그와 검색 로그를 합칩니다.
    state["log"] = new_log


    # --------------------------------------------------------
    # 2. 검색 성공 여부에 따라 분기
    # --------------------------------------------------------

    # 검색에 성공하면 답변을 생성합니다.
    if state["retrieval_ok"]:
        result = generate_node(state)

    # 검색에 실패하면 안내 메시지를 만듭니다.
    else:
        result = fallback_node(state)


    # --------------------------------------------------------
    # 3. 두 번째 Node의 결과를 State에 추가
    # --------------------------------------------------------

    # 지금까지의 로그와 새로운 로그를 합칩니다.
    new_log = state["log"] + result["log"]

    # Node 실행 결과를 State에 추가합니다.
    state.update(result)

    # 누적된 로그를 다시 State에 저장합니다.
    state["log"] = new_log

    # 최종 State를 반환합니다.
    return state


# ============================================================
# 전체 흐름 테스트
# ============================================================

# 테스트할 질문을 준비합니다.
questions = [
    "환불은 며칠 이내인가요?",
    "대표이사 이름은?",
]


# 질문을 하나씩 실행합니다.
for question in questions:

    # 전체 흐름을 실행합니다.
    result = manual_run(question)

    # 질문과 답변을 출력합니다.
    print("Q:", question)
    print("A:", result["answer"][:80])

    # 어떤 Node가 실행되었는지 로그를 출력합니다.
    for log in result["log"]:
        print("   ", log)

    # 질문별 결과를 구분합니다.
    print("-" * 55)