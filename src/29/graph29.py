# ============================================================
# Graph 29
# ============================================================
# 질문을 먼저 분류한 뒤, 질문 유형에 맞는 노드로 보냅니다.
# 문서 질문은 검색 → 생성 → 검증 과정을 거치고,
# 인사말·계산·범위 밖 질문은 바로 답변한 뒤 종료합니다.
# ============================================================


import os
import sys


# ============================================================
# 다른 차시의 파일을 import하기 위한 경로 설정
# ============================================================

# 현재 graph 파일이 있는 폴더입니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 각 차시의 폴더 경로를 추가합니다.
sys.path.insert(0, os.path.join(BASE_DIR, "..", "18"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "19"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "24"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "25"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "26"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "29"))


# ============================================================
# 필요한 모듈 가져오기
# ============================================================

from langgraph.graph import StateGraph, START, END

from graph_state import RAGState
from graph_state2 import make_initial_state

from classifier import classifier_node
from intents import (
    greeting_node,
    calc_node,
    scope_node,
)

from retriever import retriever_node
from generator import generator_node
from verifier import verifier_node
from fallback import fallback_node

import config


# config.py에 값이 없으면 기본값으로 2를 사용합니다.
MAX_RETRY = getattr(config, "MAX_RETRY", 2)


# ============================================================
# 재시도 횟수 증가 노드
# ============================================================

def bump_node(state):
    """
    답변을 다시 생성하기 전에 재시도 횟수를 1 증가시킵니다.
    """

    # 현재 재시도 횟수에 1을 더합니다.
    retries = state.get("retries", 0) + 1

    # 변경할 State 값만 반환합니다.
    return {
        "retries": retries,
        "log": [f"재시도 {retries}회차"],
    }


# ============================================================
# 질문 종류에 따른 라우팅 함수
# ============================================================

def route_intent(state):
    """
    질문 분류 결과에 따라 다음 노드를 결정합니다.
    """

    # intent가 없으면 문서 질문으로 처리합니다.
    intent = state.get("intent", "document")

    # 인사말이면 greeting 노드로 이동합니다.
    if intent == "greeting":
        return "greeting"

    # 계산식이면 calc 노드로 이동합니다.
    if intent == "calc":
        return "calc"

    # 범위 밖 질문이면 scope 노드로 이동합니다.
    if intent == "scope":
        return "scope"

    # 나머지는 모두 문서 질문으로 처리합니다.
    return "document"


# ============================================================
# 검색 결과에 따른 라우팅 함수
# ============================================================

def route_retrieve(state):
    """
    검색 성공 여부에 따라 다음 노드를 결정합니다.
    """

    # 검색 결과가 있으면 답변 생성으로 이동합니다.
    if state.get("retrieval_ok"):
        return "success"

    # 검색 결과가 없으면 fallback으로 이동합니다.
    return "fail"


# ============================================================
# 검증 결과에 따른 라우팅 함수
# ============================================================

def route_verify(state):
    """
    답변 검증 결과에 따라 종료 또는 재생성을 결정합니다.
    """

    # 검증 결과를 가져옵니다.
    grade = state.get("grade", "retry")

    # 답변이 정상이라면 그래프를 종료합니다.
    if grade == "pass":
        return "done"

    # Verifier가 포기하라고 판단하면 fallback으로 보냅니다.
    if grade == "giveup":
        return "giveup"

    # 최대 재시도 횟수를 넘으면 fallback으로 보냅니다.
    retries = state.get("retries", 0)

    if retries >= MAX_RETRY:
        return "giveup"

    # 아직 재시도할 수 있으면 다시 생성합니다.
    return "retry"


# ============================================================
# 그래프 만들기
# ============================================================

def build_graph():
    """
    LangGraph의 노드와 연결 관계를 구성합니다.
    """

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # --------------------------------------------------------
    # 1. 노드 등록
    # --------------------------------------------------------

    # 질문 분류 노드입니다.
    graph.add_node("classify", classifier_node)

    # 단축 처리 노드입니다.
    graph.add_node("greeting", greeting_node)
    graph.add_node("calc", calc_node)
    graph.add_node("scope", scope_node)

    # 기존 RAG 처리 노드입니다.
    graph.add_node("retrieve", retriever_node)
    graph.add_node("generate", generator_node)
    graph.add_node("verify", verifier_node)

    # 재시도와 실패 처리 노드입니다.
    graph.add_node("bump", bump_node)
    graph.add_node("fallback", fallback_node)

    # --------------------------------------------------------
    # 2. 그래프 시작점
    # --------------------------------------------------------

    # 모든 질문은 먼저 분류 노드로 이동합니다.
    graph.add_edge(START, "classify")

    # --------------------------------------------------------
    # 3. 질문 유형에 따른 분기
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "greeting": "greeting",
            "calc": "calc",
            "scope": "scope",
            "document": "retrieve",
        },
    )

    # --------------------------------------------------------
    # 4. 단축 처리 노드는 바로 종료
    # --------------------------------------------------------

    # 인사말 답변 후 종료합니다.
    graph.add_edge("greeting", END)

    # 계산 결과 반환 후 종료합니다.
    graph.add_edge("calc", END)

    # 범위 밖 안내 후 종료합니다.
    graph.add_edge("scope", END)

    # --------------------------------------------------------
    # 5. 검색 결과에 따른 분기
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "retrieve",
        route_retrieve,
        {
            "success": "generate",
            "fail": "fallback",
        },
    )

    # --------------------------------------------------------
    # 6. 답변 생성 후 검증
    # --------------------------------------------------------

    # 답변 생성이 끝나면 검증 노드로 이동합니다.
    graph.add_edge("generate", "verify")

    # --------------------------------------------------------
    # 7. 검증 결과에 따른 분기
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "verify",
        route_verify,
        {
            "done": END,
            "retry": "bump",
            "giveup": "fallback",
        },
    )

    # --------------------------------------------------------
    # 8. 재시도 흐름
    # --------------------------------------------------------

    # 재시도 횟수를 증가시킨 뒤 다시 답변을 생성합니다.
    graph.add_edge("bump", "generate")

    # --------------------------------------------------------
    # 9. 실패 처리
    # --------------------------------------------------------

    # fallback 안내 후 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 완성된 그래프를 반환합니다.
    return graph.compile()


# ============================================================
# 그래프 생성
# ============================================================

app = build_graph()


# ============================================================
# 질문 실행 함수
# ============================================================

def ask(question):
    """
    질문 하나를 그래프에 전달하고 최종 결과를 반환합니다.
    """

    # 질문을 초기 State로 변환합니다.
    state = make_initial_state(question)

    # 그래프를 실행합니다.
    result = app.invoke(
        state,
        {
            "recursion_limit": 25,
        },
    )

    # 최종 State를 반환합니다.
    return result


# ============================================================
# 직접 실행 테스트
# ============================================================

if __name__ == "__main__":

    # 테스트할 질문 목록입니다.
    questions = [
        "안녕하세요",
        "10 + 20",
        "오늘 날씨 어때요?",
        "환불은 며칠 이내에 신청해야 하나요?",
        "대표이사가 누구인가요?",
    ]

    # 질문을 하나씩 실행합니다.
    for question in questions:

        print("\n===================================")
        print(f"질문: {question}")

        # 그래프를 실행합니다.
        result = ask(question)

        # 최종 답변을 출력합니다.
        print(f"답변: {result.get('answer', '')}")

        # 질문 분류 결과를 출력합니다.
        print(f"분류: {result.get('intent', '')}")

        # 검증 결과를 출력합니다.
        print(f"판정: {result.get('grade', '')}")

        # 재시도 횟수를 출력합니다.
        print(f"재시도: {result.get('retries', 0)}")