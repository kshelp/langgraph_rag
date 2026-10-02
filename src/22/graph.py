# 이 파일은 LangGraph로 RAG의 검색·답변 생성·fallback 흐름을 연결합니다.
# 외부에서는 ask() 함수에 질문만 전달하면 최종 답변을 받을 수 있습니다.

import os
import sys


# ==================================================
# 다른 차시 폴더의 모듈을 가져오기 위한 설정
# ==================================================

# 현재 graph.py 파일이 있는 폴더 경로입니다.
CURRENT_DIR = os.path.dirname(__file__)

# 18차시 폴더에서 graph_state.py를 찾도록 설정합니다.
PATH_18 = os.path.join(CURRENT_DIR, "..", "18")
sys.path.insert(0, PATH_18)

# 19차시 폴더에서 나머지 노드 파일을 찾도록 설정합니다.
PATH_19 = os.path.join(CURRENT_DIR, "..", "19")
sys.path.insert(0, PATH_19)


# ==================================================
# 필요한 모듈 가져오기
# ==================================================

# LangGraph 그래프를 만드는 도구입니다.
from langgraph.graph import StateGraph, START, END

# 그래프가 사용할 상태 형식입니다.
from graph_state import RAGState

# 질문을 초기 상태로 바꾸는 함수입니다.
from graph_state2 import make_initial_state

# 각 작업을 담당하는 노드 함수입니다.
from retrieve import retrieve_node
from generate import generate_node
from fallback import fallback_node


# ==================================================
# 검색 결과에 따라 다음 노드를 결정하는 함수
# ==================================================
def route_after_retrieve(state):
    """
    검색 결과가 있으면 generate로 이동합니다.
    검색 결과가 없으면 fallback으로 이동합니다.
    """

    # 검색에 성공했는지 확인합니다.
    retrieval_ok = state["retrieval_ok"]

    # 검색 성공이면 답변 생성 노드로 이동합니다.
    if retrieval_ok:
        return "generate"

    # 검색 실패이면 fallback 노드로 이동합니다.
    return "fallback"


# ==================================================
# RAG 그래프를 만드는 함수
# ==================================================
def build_graph():
    """
    검색, 답변 생성, fallback 노드를 하나의 그래프로 연결합니다.
    """

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # 검색 노드를 등록합니다.
    graph.add_node("retrieve", retrieve_node)

    # 답변 생성 노드를 등록합니다.
    graph.add_node("generate", generate_node)

    # 검색 실패 처리 노드를 등록합니다.
    graph.add_node("fallback", fallback_node)

    # 그래프가 시작되면 retrieve부터 실행합니다.
    graph.add_edge(START, "retrieve")

    # 검색 결과에 따라 다음 노드를 결정합니다.
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            # 검색 성공 시 generate를 실행합니다.
            "generate": "generate",

            # 검색 실패 시 fallback을 실행합니다.
            "fallback": "fallback",
        },
    )

    # 답변 생성이 끝나면 그래프를 종료합니다.
    graph.add_edge("generate", END)

    # fallback 처리가 끝나면 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 그래프를 실제로 실행할 수 있는 형태로 완성합니다.
    return graph.compile()


# 프로그램이 시작될 때 그래프를 한 번 만듭니다.
app = build_graph()


# ==================================================
# 외부에서 사용하는 대표 함수
# ==================================================
def ask(question):
    """
    질문을 그래프에 전달하고 최종 결과를 반환합니다.
    """

    # 질문을 LangGraph가 사용할 초기 상태로 만듭니다.
    initial_state = make_initial_state(question)

    # 초기 상태를 그래프에 전달해 실행합니다.
    result = app.invoke(initial_state)

    # 출처 정보를 저장할 빈 리스트를 만듭니다.
    sources = []

    # 검색된 문서를 하나씩 확인합니다.
    for document in result.get("documents", []):
        source = {
            "file": document.metadata["filename"],
            "page": document.metadata["page_no"],
        }

        # 파일명과 페이지 정보를 출처 목록에 추가합니다.
        sources.append(source)

    # 외부 프로그램에서 사용할 결과를 반환합니다.
    return {
        # 최종 답변입니다.
        "answer": result["answer"],

        # 답변 생성에 사용된 문서 출처입니다.
        "sources": sources,

        # 답변에 인용이 포함되었는지 나타냅니다.
        "cited": result.get("has_citation", False),

        # 그래프가 실행된 순서 기록입니다.
        "log": result.get("log", []),

        # 함수가 정상적으로 실행되었음을 나타냅니다.
        "ok": True,
    }


# ==================================================
# 이 파일을 직접 실행했을 때의 동작
# ==================================================
if __name__ == "__main__":
    # 그래프의 노드와 연결 구조를 화면에 출력합니다.
    print(app.get_graph().draw_ascii())