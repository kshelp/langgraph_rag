import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "18"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "19"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "25"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "26"))

from langgraph.graph import StateGraph, START, END

from graph_state import RAGState
from graph_state2 import make_initial_state

from retriever import retriever_node
from generator import generator_node
from verifier import verifier_node
from fallback import fallback_node

import config


MAX_RETRY = getattr(config, "MAX_RETRY", 2)


# ===================================================
# 재시도 횟수 증가 노드
# ===================================================

def bump_node(state: RAGState) -> dict:
    n = state.get("retries", 0) + 1

    return {
        "retries": n,
        "log": [f"재시도 {n}회차 진입"],
    }


# ===================================================
# 라우팅 함수
# ===================================================

def route_after_retrieve(state: RAGState) -> str:
    return "ok" if state.get("retrieval_ok") else "empty"


def route_after_verify(state: RAGState) -> str:
    grade = state.get("grade", "retry")

    # 검증 통과
    if grade == "pass":
        return "done"

    # Verifier가 명시적으로 포기
    if grade == "giveup":
        return "giveup"

    # 최대 재시도 횟수 초과
    if state.get("retries", 0) >= MAX_RETRY:
        return "giveup"

    # 그 외에는 재생성
    return "regenerate"


# ===================================================
# 그래프 조립
# ===================================================

def build_graph():
    g = StateGraph(RAGState)

    # 노드 등록
    g.add_node("retrieve", retriever_node)
    g.add_node("generate", generator_node)
    g.add_node("verify", verifier_node)
    g.add_node("bump", bump_node)
    g.add_node("fallback", fallback_node)

    # 시작 → 검색
    g.add_edge(START, "retrieve")

    # 검색 결과에 따른 분기
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "ok": "generate",
            "empty": "fallback",
        },
    )

    # 생성 → 검증
    g.add_edge("generate", "verify")

    # 검증 결과에 따른 분기
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "done": END,
            "giveup": "fallback",
            "regenerate": "bump",
        },
    )

    # 재시도 횟수 증가 → 다시 생성
    g.add_edge("bump", "generate")

    # 검색 실패 또는 검증 포기
    g.add_edge("fallback", END)

    # 30차시에서 rewrite 및 재검색 루프 추가
    return g.compile()


app = build_graph()

def ask(question: str, verbose: bool = False) -> dict:     
    final = app.invoke(make_initial_state(question),
                       {"recursion_limit": 25})
    
    if verbose:
        print(f"\nQ: {question}")
        for line in final.get("log", []):
            print(f"   · {line}")
            
    return {
        "answer":   final["answer"],
        "sources":  [{"file": d.metadata.get("filename"),
                      "page": d.metadata.get("page_no")}
                     for d in final.get("documents", [])],
        "grade":    final.get("grade", ""),
        "reason":   final.get("reason", ""),
        "retries":  final.get("retries", 0),
        "verified": final.get("grade") == "pass",
        "log":      final.get("log", []),
        "ok":       True,
    }
    
if __name__ == "__main__":
    print(app.get_graph().draw_ascii())
    for q in ["환불은 며칠 이내에 신청해야 하나요?",
              "대표이사가 누구인가요?"]:
        r = ask(q, verbose=True)
        print(f"   A: {r['answer'][:70]}")
        print(f"   판정={r['grade']} 재시도={r['retries']}회 "
              f"검증통과={r['verified']}"
        )