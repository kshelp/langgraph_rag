# 이 파일은 Retriever, Generator, Fallback Node를 하나의 LangGraph로 연결합니다.
# 검색 결과가 있으면 답변을 생성하고, 없으면 안내 메시지를 반환합니다.

import os
import sys

# 현재 파일이 있는 폴더의 경로입니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 18차시와 19차시의 모듈을 사용할 수 있도록 경로를 추가합니다.
sys.path.insert(
    0,
    os.path.join(CURRENT_DIR, "..", "18")
)

sys.path.insert(
    0,
    os.path.join(CURRENT_DIR, "..", "19")
)


# LangGraph 그래프를 만들 때 사용하는 클래스와 상수입니다.
from langgraph.graph import StateGraph, START, END

# 그래프에서 사용할 State 구조입니다.
from graph_state import RAGState

# 질문을 초기 State 형태로 바꾸는 함수입니다.
from graph_state2 import make_initial_state

# 24차시에서 만든 검색 노드입니다.
from retriever import retriever_node

# 답변을 생성하는 노드입니다.
from generate import generate_node

# 검색 결과가 없을 때 실행하는 노드입니다.
from fallback import fallback_node


# ==================================================
# 검색 결과에 따라 다음 노드를 결정합니다.
# ==================================================
def route_after_retrieve(state: RAGState) -> str:
    # 검색에 성공했으면 generate로 이동합니다.
    if state.get("retrieval_ok"):
        return "ok"

    # 검색에 실패했으면 fallback으로 이동합니다.
    return "empty"


# ==================================================
# LangGraph를 조립합니다.
# ==================================================
def build_graph():
    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # 검색 노드를 등록합니다.
    graph.add_node("retrieve", retriever_node)

    # 답변 생성 노드를 등록합니다.
    graph.add_node("generate", generate_node)

    # 검색 실패 처리 노드를 등록합니다.
    graph.add_node("fallback", fallback_node)

    # 그래프가 시작되면 검색부터 실행합니다.
    graph.add_edge(START, "retrieve")

    # 검색 결과에 따라 다음 노드를 선택합니다.
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "ok": "generate",
            "empty": "fallback",
        }
    )

    # 답변 생성이 끝나면 그래프를 종료합니다.
    graph.add_edge("generate", END)

    # fallback 처리가 끝나면 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 나중에 다음 노드를 추가할 수 있습니다.
    # 26차시: verify 노드
    # 27차시: bump 노드와 재생성 루프
    # 30차시: rewrite 노드와 재검색 루프

    # 그래프를 실행할 수 있는 형태로 컴파일합니다.
    return graph.compile()


# 그래프를 한 번 만들어 앱으로 사용합니다.
app = build_graph()


# ==================================================
# 질문 하나를 그래프에 전달하는 함수입니다.
# ==================================================
def ask(question: str) -> dict:
    # 질문을 초기 State로 변환합니다.
    initial_state = make_initial_state(question)

    # 초기 State를 그래프에 전달합니다.
    final_state = app.invoke(
        initial_state,
        {
            "recursion_limit": 20
        }
    )

    # 최종 State에서 필요한 결과만 골라 반환합니다.
    return {
        "answer": final_state.get("answer", ""),

        "sources": [
            {
                "file": document.metadata.get("filename", ""),
                "page": document.metadata.get("page_no", 0),
            }
            for document in final_state.get("documents", [])
        ],

        "cited": final_state.get("has_citation", False),
        "log": final_state.get("log", []),
        "ok": final_state.get("retrieval_ok", False),
        "fail_reason": final_state.get("fail_reason", ""),
    }


# ==================================================
# 이 파일을 직접 실행할 때만 테스트합니다.
# ==================================================
if __name__ == "__main__":

    # 테스트할 질문 목록입니다.
    questions = [
        "환불은 며칠 이내인가요?",
        "반품하고 싶은데 언제까지?",
    ]

    # 질문을 하나씩 테스트합니다.
    for question in questions:

        # ask 함수를 사용해 그래프를 실행합니다.
        result = ask(question)

        # 질문을 출력합니다.
        print(f"\nQ: {question}")

        # 검색 성공 여부와 실패 이유를 출력합니다.
        print(
            f"   검색 성공: {result['ok']} "
            f"사유: {result['fail_reason'] or '-'}"
        )

        # 최종 답변의 앞부분만 출력합니다.
        print(
            f"   답변: {result['answer'][:60]}..."
        )