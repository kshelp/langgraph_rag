# ============================================================
# Graph 31 - 방어적 코딩 + 운영 로그
# 기존 RAG 노드를 safe_node로 안전하게 실행합니다.
# 오류가 발생해도 그래프를 중단하지 않고 로그를 남깁니다.
# ============================================================

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 필요한 차시의 모듈 경로를 추가합니다.
# ============================================================

BASE_DIR = os.path.dirname(__file__)

for folder in ["18", "19", "24", "25", "26", "29", "30"]:
    path = os.path.join(BASE_DIR, "..", folder)
    sys.path.insert(0, path)


# ============================================================
# 필요한 모듈을 가져옵니다.
# ============================================================

from langgraph.graph import StateGraph, START, END

from graph_state2 import RAGState, make_initial_state

from retriever import retriever_node
from generator import generator_node
from verifier import verifier_node
from fallback import fallback_node

from classifier import classifier_node
from intents import greeting_node, calc_node, scope_node
from rewriter import rewrite_node

from safe import safe_node
from logger import log_query

import config


# ============================================================
# 최대 재시도 / 재작성 횟수
# ============================================================

MAX_RETRY = getattr(config, "MAX_RETRY", 2)
MAX_REWRITE = getattr(config, "MAX_REWRITE", 2)


# ============================================================
# 안전한 Retriever
# 검색 중 오류가 발생해도 그래프가 중단되지 않게 합니다.
# ============================================================

def safe_retriever_node(state):

    # retriever_node를 safe_node를 통해 실행합니다.
    result = safe_node(retriever_node, state)

    # 검색 중 오류가 발생한 경우입니다.
    if result.get("node_error"):

        # 검색 결과를 비웁니다.
        result["documents"] = []

        # 검색 실패 상태로 변경합니다.
        result["retrieval_ok"] = False

        # 검색 시스템 오류임을 기록합니다.
        result["fail_reason"] = "search_error"

    return result


# ============================================================
# 안전한 Generator
# 답변 생성 중 오류가 발생해도 그래프를 계속 실행합니다.
# ============================================================

def safe_generator_node(state):

    # generator_node를 safe_node를 통해 실행합니다.
    result = safe_node(generator_node, state)

    # 답변 생성 중 오류가 발생한 경우입니다.
    if result.get("node_error"):

        # 생성된 답변이 없음을 표시합니다.
        result["answer"] = ""

        # 정상적인 답변 생성에 실패했음을 표시합니다.
        result["insufficient"] = True

        # 생성 오류를 기록합니다.
        result["gen_error"] = "unknown"

    return result


# ============================================================
# 안전한 Verifier
# 검증 중 오류가 발생해도 전체 서비스를 중단하지 않습니다.
# ============================================================

def safe_verifier_node(state):

    # verifier_node를 safe_node를 통해 실행합니다.
    result = safe_node(verifier_node, state)

    # 검증 중 오류가 발생한 경우입니다.
    if result.get("node_error"):

        # 그래프는 종료할 수 있도록 pass로 처리합니다.
        result["grade"] = "pass"

        # 실제 검증에는 실패했다는 이유를 기록합니다.
        result["reason"] = "검증 오류 - 통과 처리"

        # 정상 검증이 아니라는 것을 표시합니다.
        result["verified"] = False

    return result


# ============================================================
# 재생성 횟수를 1 증가시킵니다.
# ============================================================

def bump_node(state):

    retries = state.get("retries", 0)
    retries = retries + 1

    return {
        "retries": retries,
        "log": [f"재시도 {retries}회차"]
    }


# ============================================================
# 질문 종류에 따라 다음 노드를 결정합니다.
# ============================================================

def route_by_intent(state):

    intent = state.get("intent", "document")

    # 인사 질문입니다.
    if intent == "greeting":
        return "greeting"

    # 계산 질문입니다.
    if intent == "calc":
        return "calc"

    # 서비스 범위 밖의 질문입니다.
    if intent == "scope":
        return "scope"

    # 나머지는 문서 검색으로 보냅니다.
    return "document"


# ============================================================
# 검색 결과에 따라 다음 노드를 결정합니다.
# ============================================================

def route_after_retrieve(state):

    # 검색에 성공했으면 답변을 생성합니다.
    if state.get("retrieval_ok"):
        return "ok"

    # 최대 재작성 횟수에 도달하면 포기합니다.
    rewrites = state.get("rewrites", 0)

    if rewrites >= MAX_REWRITE:
        return "giveup"

    # 검색 실패 이유를 가져옵니다.
    fail_reason = state.get("fail_reason", "")

    # 자료의 관련도가 낮으면 질문을 바꿔 다시 검색합니다.
    if fail_reason == "low_score":
        return "research"

    # 실패 이유가 없으면 재검색을 시도합니다.
    if fail_reason == "":
        return "research"

    # 시스템 오류 등은 재검색하지 않고 포기합니다.
    return "giveup"


# ============================================================
# 검증 결과에 따라 다음 노드를 결정합니다.
# ============================================================

def route_after_verify(state):

    grade = state.get("grade", "retry")

    # 검증에 통과했으면 종료합니다.
    if grade == "pass":
        return "done"

    # 더 이상 처리할 수 없으면 포기합니다.
    if grade == "giveup":
        return "giveup"

    # 자료가 부족하면 질문을 바꿔 다시 검색합니다.
    if grade == "research":

        rewrites = state.get("rewrites", 0)

        # 최대 재작성 횟수에 도달했는지 확인합니다.
        if rewrites >= MAX_REWRITE:
            return "giveup"

        return "research"

    # 현재 재생성 횟수를 가져옵니다.
    retries = state.get("retries", 0)

    # 최대 재생성 횟수에 도달하면 포기합니다.
    if retries >= MAX_RETRY:
        return "giveup"

    # 같은 자료로 답변을 다시 생성합니다.
    return "regenerate"


# ============================================================
# 질문 재작성 결과에 따라 다시 검색할지 결정합니다.
# ============================================================

def route_after_rewrite(state):

    # 새롭게 작성된 검색어를 가져옵니다.
    query = state.get("query", "")

    # 지금까지 사용한 검색어를 가져옵니다.
    tried_queries = state.get("tried_queries", [])

    # 검색어 생성에 실패하면 포기합니다.
    if not query:
        return "giveup"

    # 새로운 검색어가 등록되었으면 다시 검색합니다.
    if query in tried_queries:
        return "retry_search"

    # 정상적인 재작성이 아니면 포기합니다.
    return "giveup"


# ============================================================
# LangGraph를 구성합니다.
# ============================================================

def build_graph():

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)


    # --------------------------------------------------------
    # 질문 분류 노드
    # --------------------------------------------------------

    graph.add_node("classify", classifier_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("calc", calc_node)
    graph.add_node("scope", scope_node)


    # --------------------------------------------------------
    # RAG 노드
    # safe_node를 적용한 안전한 노드를 등록합니다.
    # --------------------------------------------------------

    graph.add_node("retrieve", safe_retriever_node)
    graph.add_node("generate", safe_generator_node)
    graph.add_node("verify", safe_verifier_node)


    # --------------------------------------------------------
    # 재시도 / 재검색 / 실패 처리 노드
    # --------------------------------------------------------

    graph.add_node("bump", bump_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("fallback", fallback_node)


    # --------------------------------------------------------
    # START → classify
    # --------------------------------------------------------

    graph.add_edge(START, "classify")


    # --------------------------------------------------------
    # 질문 종류에 따라 분기합니다.
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


    # 인사, 계산, 범위 밖 질문은 바로 종료합니다.
    graph.add_edge("greeting", END)
    graph.add_edge("calc", END)
    graph.add_edge("scope", END)


    # --------------------------------------------------------
    # 검색 결과에 따라 분기합니다.
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
    # 답변을 생성한 후 검증합니다.
    # --------------------------------------------------------

    graph.add_edge("generate", "verify")


    # --------------------------------------------------------
    # 검증 결과에 따라 분기합니다.
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
    # 답변 재생성: bump → generate
    # --------------------------------------------------------

    graph.add_edge("bump", "generate")


    # --------------------------------------------------------
    # 질문 재작성 후 다시 검색하거나 포기합니다.
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "retry_search": "retrieve",
            "giveup": "fallback"
        }
    )


    # --------------------------------------------------------
    # 실패 안내 후 종료합니다.
    # --------------------------------------------------------

    graph.add_edge("fallback", END)


    # 완성된 그래프를 실행 가능한 형태로 만듭니다.
    return graph.compile()


# ============================================================
# 그래프를 생성합니다.
# ============================================================

app = build_graph()


# ============================================================
# 사용자 질문을 그래프에 전달합니다.
# 실행 결과와 처리 시간을 운영 로그에 저장합니다.
# ============================================================

def ask(question, verbose=False):

    # 실행 시작 시간을 기록합니다.
    start = time.time()

    # 질문으로 초기 State를 만듭니다.
    state = make_initial_state(question)

    # LangGraph를 실행합니다.
    final = app.invoke(
        state,
        {"recursion_limit": 25}
    )

    # 전체 실행 시간을 계산합니다.
    elapsed = time.time() - start

    # 질문 처리 결과를 운영 로그에 저장합니다.
    log_query(question, final, elapsed)


    # --------------------------------------------------------
    # 상세 실행 과정을 출력합니다.
    # --------------------------------------------------------

    if verbose:

        print(f"\nQ: {question}")

        for line in final.get("log", []):
            print("  ·", line)


    # --------------------------------------------------------
    # 검색 문서의 출처를 정리합니다.
    # --------------------------------------------------------

    sources = []

    documents = final.get("documents", [])

    for doc in documents:

        source = {
            "file": doc.metadata.get("filename"),
            "page": doc.metadata.get("page_no")
        }

        sources.append(source)


    # --------------------------------------------------------
    # 사용자에게 필요한 결과를 반환합니다.
    # --------------------------------------------------------

    result = {
        "answer": final.get("answer", ""),
        "sources": sources,
        "intent": final.get("intent", ""),
        "grade": final.get("grade", ""),
        "reason": final.get("reason", ""),
        "retries": final.get("retries", 0),
        "rewrites": final.get("rewrites", 0),
        "tried_queries": final.get("tried_queries", []),
        "node_error": final.get("node_error", ""),
        "fail_reason": final.get("fail_reason", ""),
        "fallback_kind": final.get("fallback_kind", ""),
        "log": final.get("log", []),
        "ok": True
    }

    # 정상적인 검증 통과 여부를 계산합니다.
    result["verified"] = False

    if final.get("grade") == "pass":

        # 노드 오류가 없을 때만 정상 검증으로 봅니다.
        if not final.get("node_error"):
            result["verified"] = True

    return result


# ============================================================
# 직접 실행할 때 테스트합니다.
# ============================================================

if __name__ == "__main__":

    # 현재 그래프 구조를 출력합니다.
    print(app.get_graph().draw_ascii())

    # 여러 종류의 질문을 준비합니다.
    questions = [
        "안녕하세요",
        "10 + 20",
        "오늘 날씨 어때요?",
        "환불은 며칠 이내에 신청해야 하나요?",
        "대표이사가 누구인가요?"
    ]

    # 질문을 하나씩 실행합니다.
    for question in questions:

        result = ask(
            question,
            verbose=True
        )

        print("A:", result["answer"])
        print("의도:", result["intent"])
        print("판정:", result["grade"])
        print("재시도:", result["retries"])
        print("재작성:", result["rewrites"])

        # 노드 오류가 있으면 출력합니다.
        if result["node_error"]:
            print("노드 오류:", result["node_error"])

        print()