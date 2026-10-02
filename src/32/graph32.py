# ============================================================
# graph32.py
# 지금까지 만든 RAG 노드들을 하나의 LangGraph로 연결합니다.
# 질문 종류와 처리 결과에 따라 적절한 노드로 이동합니다.
# ============================================================

import os
import sys

from langgraph.graph import StateGraph, START, END


# ============================================================
# 이전 차시의 모듈을 사용할 수 있도록 경로 추가
# ============================================================

BASE_DIR = os.path.dirname(__file__)

folders = ["18", "19", "24", "25", "26", "29", "30", "31"]

for folder in folders:
    path = os.path.join(BASE_DIR, "..", folder)
    sys.path.insert(0, path)


# ============================================================
# 필요한 모듈 불러오기
# ============================================================

from graph_state2 import RAGState, make_initial_state

from classifier import classifier_node
from intents import greeting_node, calc_node, scope_node

from retriever import retriever_node
from generator import generator_node
from verifier import verifier_node
from rewriter import rewrite_node
from fallback import fallback_node

import config


# ============================================================
# 최대 반복 횟수
# ============================================================

# 같은 자료로 답변을 다시 만드는 최대 횟수
MAX_RETRY = getattr(config, "MAX_RETRY", 2)

# 질문을 고쳐서 다시 검색하는 최대 횟수
MAX_REWRITE = getattr(config, "MAX_REWRITE", 2)


# ============================================================
# 재생성 횟수 증가 노드
# ============================================================

def bump_node(state):
    """답변 재생성 횟수를 1 증가시킵니다."""

    retries = state.get("retries", 0)
    retries = retries + 1

    return {
        "retries": retries,
        "log": [f"재생성 {retries}회차"]
    }


# ============================================================
# 1. 질문 종류에 따른 이동 경로 결정
# ============================================================

def route_by_intent(state):
    """분류된 질문 종류에 따라 다음 경로를 결정합니다."""

    intent = state.get("intent", "document")

    if intent == "greeting":
        return "greeting"

    if intent == "calc":
        return "calc"

    if intent == "scope":
        return "scope"

    # 나머지 질문은 문서 검색으로 이동
    return "document"


# ============================================================
# 2. 검색 결과에 따른 이동 경로 결정
# ============================================================

def route_after_retrieve(state):
    """검색 성공 여부에 따라 생성, 재검색, 포기를 결정합니다."""

    # 검색에 성공하면 답변 생성
    if state.get("retrieval_ok"):
        return "ok"

    # 재검색 횟수를 모두 사용했으면 포기
    rewrites = state.get("rewrites", 0)

    if rewrites >= MAX_REWRITE:
        return "giveup"

    # 검색 시스템 자체의 문제라면 다시 검색하지 않음
    fail_reason = state.get("fail_reason", "")

    if fail_reason == "search_error":
        return "giveup"

    if fail_reason == "no_result":
        return "giveup"

    if fail_reason == "empty_query":
        return "giveup"

    # 검색 결과의 품질이 낮으면 질문을 고쳐서 재검색
    return "research"


# ============================================================
# 3. 답변 검증 결과에 따른 이동 경로 결정
# ============================================================

def route_after_verify(state):
    """검증 결과에 따라 종료, 재생성, 재검색을 결정합니다."""

    grade = state.get("grade", "retry")

    # 답변이 정상이면 종료
    if grade == "pass":
        return "done"

    # 더 이상 처리할 수 없으면 포기
    if grade == "giveup":
        return "giveup"

    # 검색 자료가 부족하면 질문을 고쳐서 재검색
    if grade == "research":

        rewrites = state.get("rewrites", 0)

        if rewrites >= MAX_REWRITE:
            return "giveup"

        return "research"

    # 답변에 문제가 있으면 같은 자료로 다시 생성
    retries = state.get("retries", 0)

    if retries < MAX_RETRY:
        return "regenerate"

    # 재생성 횟수를 모두 사용했으면 포기
    return "giveup"


# ============================================================
# 4. 질문 재작성 결과에 따른 이동 경로 결정
# ============================================================

def route_after_rewrite(state):
    """새 검색어가 만들어졌으면 다시 검색합니다."""

    query = state.get("query", "")

    if query:
        return "retry_search"

    return "giveup"


# ============================================================
# 전체 LangGraph 만들기
# ============================================================

def build_graph():
    """모든 노드를 등록하고 실행 순서를 연결합니다."""

    graph = StateGraph(RAGState)


    # --------------------------------------------------------
    # 1. 노드 등록
    # --------------------------------------------------------

    graph.add_node("classify", classifier_node)

    graph.add_node("greeting", greeting_node)
    graph.add_node("calc", calc_node)
    graph.add_node("scope", scope_node)

    graph.add_node("retrieve", retriever_node)
    graph.add_node("generate", generator_node)
    graph.add_node("verify", verifier_node)

    graph.add_node("bump", bump_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("fallback", fallback_node)


    # --------------------------------------------------------
    # 2. 시작 → 질문 분류
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "classify"
    )


    # --------------------------------------------------------
    # 3. 질문 종류에 따라 분기
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "greeting": "greeting",
            "calc": "calc",
            "scope": "scope",
            "document": "retrieve"
        }
    )


    # 인사, 계산, 범위 밖 질문은 바로 종료
    graph.add_edge("greeting", END)
    graph.add_edge("calc", END)
    graph.add_edge("scope", END)


    # --------------------------------------------------------
    # 4. 검색 결과에 따라 분기
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "ok": "generate",
            "research": "rewrite",
            "giveup": "fallback"
        }
    )


    # --------------------------------------------------------
    # 5. 답변 생성 → 검증
    # --------------------------------------------------------

    graph.add_edge(
        "generate",
        "verify"
    )


    # --------------------------------------------------------
    # 6. 검증 결과에 따라 분기
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "done": END,
            "regenerate": "bump",
            "research": "rewrite",
            "giveup": "fallback"
        }
    )


    # --------------------------------------------------------
    # 7. 재생성 루프
    # --------------------------------------------------------
    # 같은 검색 자료를 사용하여 답변만 다시 만듭니다.

    graph.add_edge(
        "bump",
        "generate"
    )


    # --------------------------------------------------------
    # 8. 재검색 루프
    # --------------------------------------------------------
    # 질문을 고친 후 문서를 다시 검색합니다.

    graph.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "retry_search": "retrieve",
            "giveup": "fallback"
        }
    )


    # --------------------------------------------------------
    # 9. 실패 안내 후 종료
    # --------------------------------------------------------

    graph.add_edge(
        "fallback",
        END
    )


    # 완성된 그래프 반환
    return graph.compile()


# ============================================================
# 그래프 생성
# ============================================================

app = build_graph()


# ============================================================
# 질문 실행 함수
# ============================================================

def ask(question):
    """질문을 받아 초기 State를 만들고 그래프를 실행합니다."""

    # 질문을 이용하여 초기 State 생성
    state = make_initial_state(question)

    # 완성된 그래프 실행
    result = app.invoke(
        state,
        {"recursion_limit": 40}
    )

    return result


# ============================================================
# 직접 실행 테스트
# ============================================================

if __name__ == "__main__":

    # 테스트 질문
    question = "환불은 며칠 이내에 신청하고 대표이사 이름은?"

    # 그래프 실행
    result = ask(question)

    # 최종 결과 확인
    print("질문   :", question)
    print("답변   :", result.get("answer"))
    print("의도   :", result.get("intent"))
    print("판정   :", result.get("grade"))
    print("재생성 :", result.get("retries", 0))
    print("재검색 :", result.get("rewrites", 0))