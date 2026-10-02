# ============================================================
# 30차시 - Query Rewrite + 재검색
# 질문을 분류하고 문서를 검색한 뒤 답변을 생성합니다.
# 검색 실패 시 검색어를 다시 작성하여 재검색합니다.
# ============================================================

import os
import sys
import warnings


# ------------------------------------------------------------
# 필요한 모듈의 경로를 설정합니다.
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

warnings.filterwarnings("ignore")

# 공통 모듈이 있는 src 폴더를 추가합니다.
sys.path.insert(0, SRC_DIR)

# 이전 차시에서 만든 모듈을 사용합니다.
for folder in ["18", "19", "24", "25", "26", "29"]:
    path = os.path.join(SRC_DIR, folder)
    sys.path.append(path)

# 현재 30차시 모듈을 가장 먼저 찾도록 합니다.
sys.path.insert(0, CURRENT_DIR)


# ------------------------------------------------------------
# 필요한 모듈을 가져옵니다.
# ------------------------------------------------------------

import config

from langgraph.graph import StateGraph, START, END

from graph_state2 import RAGState
from graph_state2 import make_initial_state

from retriever import retriever_node
from generator import generator_node
from verifier import verifier_node
from fallback import fallback_node
from classifier import classifier_node
from rewriter import rewrite_node

from intents import (
    greeting_node,
    calc_node,
    scope_node,
)


# ------------------------------------------------------------
# 최대 재생성 및 재작성 횟수를 설정합니다.
# ------------------------------------------------------------

MAX_RETRY = getattr(config, "MAX_RETRY", 2)
MAX_REWRITE = getattr(config, "MAX_REWRITE", 2)


# ============================================================
# 재생성 횟수 증가 Node
# 같은 문서로 답변을 다시 생성할 때 횟수를 증가시킵니다.
# ============================================================

def bump_node(state):

    # 현재 재생성 횟수를 1 증가시킵니다.
    retries = state.get("retries", 0) + 1

    return {
        "retries": retries,
        "log": [f"재생성 {retries}회"],
    }


# ============================================================
# 질문 유형에 따른 경로 선택
# greeting, calc, scope, document 중 하나를 선택합니다.
# ============================================================

def route_intent(state):

    # Classifier가 판단한 질문 유형을 가져옵니다.
    intent = state.get("intent", "document")

    # 인사 질문입니다.
    if intent == "greeting":
        return "greeting"

    # 계산 질문입니다.
    if intent == "calc":
        return "calc"

    # 처리 범위를 벗어난 질문입니다.
    if intent == "scope":
        return "scope"

    # 나머지는 문서 검색 질문으로 처리합니다.
    return "document"


# ============================================================
# 검색 결과에 따른 경로 선택
# 성공하면 답변을 만들고, 실패하면 검색어를 재작성합니다.
# ============================================================

def route_retrieve(state):

    # 관련 문서를 찾았으면 답변 생성으로 이동합니다.
    if state.get("retrieval_ok"):
        return "generate"

    # 최대 재작성 횟수에 도달하면 포기합니다.
    rewrites = state.get("rewrites", 0)

    if rewrites >= MAX_REWRITE:
        return "fallback"

    # 검색 점수가 낮으면 검색어를 다시 작성합니다.
    fail_reason = state.get("fail_reason", "")

    if fail_reason == "low_score":
        return "rewrite"

    # 실패 이유가 없더라도 재작성을 시도합니다.
    if fail_reason == "":
        return "rewrite"

    # 검색 오류 등은 더 진행하지 않습니다.
    return "fallback"


# ============================================================
# 답변 검증 결과에 따른 경로 선택
# 통과, 재생성, 재검색, 포기 중 하나를 선택합니다.
# ============================================================

def route_verify(state):

    # Verifier가 판단한 결과를 가져옵니다.
    grade = state.get("grade", "retry")

    # 답변이 정상이면 종료합니다.
    if grade == "pass":
        return "end"

    # 더 이상 진행할 수 없으면 fallback으로 갑니다.
    if grade == "giveup":
        return "fallback"

    # 근거가 부족하면 검색어를 바꿔 다시 검색합니다.
    if grade == "research":

        rewrites = state.get("rewrites", 0)

        if rewrites >= MAX_REWRITE:
            return "fallback"

        return "rewrite"

    # 답변만 다시 생성할 수 있는지 확인합니다.
    retries = state.get("retries", 0)

    if retries >= MAX_RETRY:
        return "fallback"

    # 같은 검색 결과를 사용하여 답변을 다시 만듭니다.
    return "retry"


# ============================================================
# LangGraph 생성
# 각 Node와 이동 경로를 연결합니다.
# ============================================================

def build_graph():

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)


    # --------------------------------------------------------
    # 1. Node 등록
    # --------------------------------------------------------

    graph.add_node("classify", classifier_node)

    graph.add_node("greeting", greeting_node)
    graph.add_node("calc", calc_node)
    graph.add_node("scope", scope_node)

    graph.add_node("retrieve", retriever_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generator_node)
    graph.add_node("verify", verifier_node)

    graph.add_node("bump", bump_node)
    graph.add_node("fallback", fallback_node)


    # --------------------------------------------------------
    # 2. 시작 → 질문 분류
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "classify"
    )


    # --------------------------------------------------------
    # 3. 질문 유형에 따라 이동합니다.
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "greeting": "greeting",
            "calc": "calc",
            "scope": "scope",
            "document": "retrieve",
        }
    )

    # 단순 질문은 바로 종료합니다.
    graph.add_edge("greeting", END)
    graph.add_edge("calc", END)
    graph.add_edge("scope", END)


    # --------------------------------------------------------
    # 4. 검색 결과에 따라 이동합니다.
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "retrieve",
        route_retrieve,
        {
            # 검색 성공 → 답변 생성
            "generate": "generate",

            # 검색 실패 → 검색어 재작성
            "rewrite": "rewrite",

            # 더 이상 처리 불가
            "fallback": "fallback",
        }
    )


    # --------------------------------------------------------
    # 5. 검색어를 바꾼 후 다시 검색합니다.
    # --------------------------------------------------------

    graph.add_edge(
        "rewrite",
        "retrieve"
    )


    # --------------------------------------------------------
    # 6. 답변을 생성한 후 검증합니다.
    # --------------------------------------------------------

    graph.add_edge(
        "generate",
        "verify"
    )


    # --------------------------------------------------------
    # 7. 검증 결과에 따라 이동합니다.
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "verify",
        route_verify,
        {
            # 검증 통과 → 종료
            "end": END,

            # 답변 문제 → 같은 근거로 다시 생성
            "retry": "bump",

            # 근거 문제 → 검색어를 바꿔 재검색
            "rewrite": "rewrite",

            # 해결할 수 없음
            "fallback": "fallback",
        }
    )


    # --------------------------------------------------------
    # 8. 재생성 횟수를 증가시킨 후 다시 생성합니다.
    # --------------------------------------------------------

    graph.add_edge(
        "bump",
        "generate"
    )


    # --------------------------------------------------------
    # 9. Fallback 처리 후 종료합니다.
    # --------------------------------------------------------

    graph.add_edge(
        "fallback",
        END
    )


    # 완성된 그래프를 실행 가능한 형태로 만듭니다.
    return graph.compile()


# ============================================================
# 그래프 컴파일
# ============================================================

app = build_graph()


# ============================================================
# 질문 실행 함수
# 질문을 받아 그래프를 실행하고 필요한 결과를 반환합니다.
# ============================================================

def ask(question, verbose=False):

    # 질문으로 초기 State를 만듭니다.
    state = make_initial_state(question)

    # LangGraph를 실행합니다.
    final = app.invoke(
        state,
        {
            "recursion_limit": 25
        }
    )


    # --------------------------------------------------------
    # 실행 과정을 화면에 출력합니다.
    # --------------------------------------------------------

    if verbose:

        print(f"\nQ: {question}")

        for line in final.get("log", []):
            print("  ·", line)


    # --------------------------------------------------------
    # 검색한 문서의 출처를 정리합니다.
    # --------------------------------------------------------

    sources = []

    documents = final.get("documents", [])

    for document in documents:

        source = {
            "file": document.metadata.get("filename"),
            "page": document.metadata.get("page_no"),
        }

        sources.append(source)


    # --------------------------------------------------------
    # 필요한 결과만 정리하여 반환합니다.
    # --------------------------------------------------------

    result = {
        "answer": final.get("answer", ""),
        "intent": final.get("intent", ""),
        "grade": final.get("grade", ""),
        "reason": final.get("reason", ""),
        "retries": final.get("retries", 0),
        "rewrites": final.get("rewrites", 0),
        "tried_queries": final.get("tried_queries", []),
        "sources": sources,
        "log": final.get("log", []),
    }

    return result


# ============================================================
# 직접 실행 테스트
# ============================================================

if __name__ == "__main__":

    # 여러 종류의 질문으로 그래프를 테스트합니다.
    questions = [
        "안녕하세요",
        "10 + 20",
        "오늘 날씨 어때요?",
        "환불은 며칠 이내에 신청해야 하나요?",
        "반품하고 싶은데 언제까지 가능해요?",
        "대표이사가 누구인가요?",
    ]


    # --------------------------------------------------------
    # 질문을 하나씩 실행합니다.
    # --------------------------------------------------------

    for question in questions:

        result = ask(
            question,
            verbose=True
        )

        # 최종 답변을 출력합니다.
        print("A:", result["answer"])

        # 재생성 횟수를 출력합니다.
        print(
            "재생성:",
            result["retries"],
            "회"
        )

        # 검색어 재작성 횟수를 출력합니다.
        print(
            "재작성:",
            result["rewrites"],
            "회"
        )

        # 검색에 사용한 검색어를 출력합니다.
        print(
            "검색어:",
            result["tried_queries"]
        )

        print("-" * 60)