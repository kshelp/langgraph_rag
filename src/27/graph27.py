# 이 파일은 Retriever, Generator, Verifier를 하나의 그래프로 연결합니다.
# 답변 검증에 실패하면 최대 2회까지 다시 생성하고, 계속 실패하면 안내 메시지를 반환합니다.


import os
import sys


# ============================================================
# 필요한 폴더를 Python이 찾을 수 있도록 등록합니다.
# ============================================================

BASE_DIR = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(BASE_DIR, "..", "18"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "19"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "24"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "25"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "26"))


# ============================================================
# LangGraph와 각 모듈을 가져옵니다.
# ============================================================

from langgraph.graph import StateGraph, START, END

from graph_state import RAGState
from graph_state2 import make_initial_state

from retriever import retriever_node
from generator import generator_node
from verifier import verifier_node
from fallback import fallback_node

import config


# 최대 재시도 횟수를 설정합니다.
MAX_RETRY = getattr(config, "MAX_RETRY", 2)


# ============================================================
# 재시도 횟수 증가 노드
# ============================================================

def bump_node(state):
    """답변 재생성 횟수를 1 증가시킵니다."""

    # 현재 재시도 횟수를 가져옵니다.
    current_retries = state.get("retries", 0)

    # 재시도 횟수를 1 증가시킵니다.
    next_retries = current_retries + 1

    # 변경할 State 값만 반환합니다.
    return {
        "retries": next_retries,
        "log": [
            f"재시도 {next_retries}회차 진입"
        ],
    }


# ============================================================
# 검색 결과에 따른 이동 결정
# ============================================================

def route_after_retrieve(state):
    """검색 성공 여부에 따라 다음 노드를 결정합니다."""

    # 검색에 성공하면 답변을 생성합니다.
    if state.get("retrieval_ok"):
        return "ok"

    # 검색에 실패하면 안내 메시지를 반환합니다.
    return "empty"


# ============================================================
# 검증 결과에 따른 이동 결정
# ============================================================

def route_after_verify(state):
    """Verifier의 판정 결과에 따라 다음 노드를 결정합니다."""

    # 검증 결과를 가져옵니다.
    grade = state.get("grade", "retry")

    # 답변이 검증되었으면 그래프를 종료합니다.
    if grade == "pass":
        return "done"

    # Verifier가 포기 판정을 내리면 fallback으로 이동합니다.
    if grade == "giveup":
        return "giveup"

    # 최대 재시도 횟수에 도달했으면 fallback으로 이동합니다.
    retries = state.get("retries", 0)

    if retries >= MAX_RETRY:
        return "giveup"

    # 아직 재시도할 수 있으면 답변을 다시 생성합니다.
    return "retry"


# ============================================================
# LangGraph 조립
# ============================================================

def build_graph():
    """RAG 처리에 필요한 노드와 연결을 구성합니다."""

    # State 구조를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # 각 처리 함수를 그래프의 노드로 등록합니다.
    graph.add_node("retrieve", retriever_node)
    graph.add_node("generate", generator_node)
    graph.add_node("verify", verifier_node)
    graph.add_node("bump", bump_node)
    graph.add_node("fallback", fallback_node)

    # 그래프가 시작되면 검색부터 실행합니다.
    graph.add_edge(START, "retrieve")

    # 검색 결과에 따라 생성 또는 fallback으로 이동합니다.
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "ok": "generate",
            "empty": "fallback",
        },
    )

    # 답변 생성이 끝나면 검증을 실행합니다.
    graph.add_edge("generate", "verify")

    # 검증 결과에 따라 종료, 재생성 또는 fallback으로 이동합니다.
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "done": END,
            "giveup": "fallback",
            "retry": "bump",
        },
    )

    # 재시도 횟수를 증가시킨 후 다시 답변을 생성합니다.
    graph.add_edge("bump", "generate")

    # fallback 처리가 끝나면 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 구성한 그래프를 실행 가능한 형태로 컴파일합니다.
    return graph.compile()


# 그래프를 한 번 만들어 여러 질문에 재사용합니다.
app = build_graph()


# ============================================================
# 외부에서 질문을 처리하는 함수
# ============================================================

def ask(question):
    """질문을 그래프에 전달하고 최종 결과를 반환합니다."""

    # 질문으로 초기 State를 만듭니다.
    initial_state = make_initial_state(question)

    # 전체 그래프를 실행합니다.
    result = app.invoke(
        initial_state,
        {
            "recursion_limit": 25
        },
    )

    # 사용자에게 필요한 결과만 정리해서 반환합니다.
    return {
        "answer": result.get("answer", ""),
        "grade": result.get("grade", ""),
        "reason": result.get("reason", ""),
        "retries": result.get("retries", 0),
    }


# ============================================================
# 직접 실행 테스트
# ============================================================

if __name__ == "__main__":

    # 테스트할 질문 목록입니다.
    questions = [
        "환불은 며칠 이내에 신청해야 하나요?",
        "대표이사가 누구인가요?",
        "환불 방법과 수수료를 알려주세요"
    ]

    # 질문을 하나씩 실행합니다.
    for question in questions:

        # 그래프에 질문을 전달합니다.
        result = ask(question)

        # 실행 결과를 출력합니다.
        print()
        print("Q:", question)
        print("A:", result["answer"])
        print("판정:", result["grade"])
        print("재시도:", result["retries"])