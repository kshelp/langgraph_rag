# 이 파일은 LangGraph로 검색, 답변 생성, fallback 흐름을 연결합니다.
# 질문은 ask() 함수로 전달하며, 그래프 실행 결과를 답변과 출처 형태로 반환합니다.

import os
import sys


# ==================================================
# 다른 차시 폴더의 모듈을 찾기 위한 설정
# ==================================================

# 현재 파일이 있는 폴더입니다.
CURRENT_DIR = os.path.dirname(__file__)

# 18차시 폴더를 모듈 검색 경로에 추가합니다.
sys.path.insert(
    0,
    os.path.join(CURRENT_DIR, "..", "18")
)

# 19차시 폴더를 모듈 검색 경로에 추가합니다.
sys.path.insert(
    0,
    os.path.join(CURRENT_DIR, "..", "19")
)


# ==================================================
# 필요한 모듈 가져오기
# ==================================================

# LangGraph 그래프를 만드는 도구입니다.
from langgraph.graph import StateGraph, START, END

# 그래프가 사용할 상태 형식입니다.
from graph_state import RAGState

# 질문을 초기 상태로 만드는 함수입니다.
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
    검색 성공 여부를 확인하고 다음 노드를 결정합니다.
    """

    # 검색 성공이면 generate 노드로 이동합니다.
    if state.get("retrieval_ok"):
        return "generate"

    # 검색 실패이면 fallback 노드로 이동합니다.
    return "fallback"


# ==================================================
# LangGraph를 만드는 함수
# ==================================================
def build_graph():
    """
    retrieve, generate, fallback 노드를 연결합니다.
    """

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # 검색 노드를 등록합니다.
    graph.add_node("retrieve", retrieve_node)

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
            # 검색 성공 시 답변을 생성합니다.
            "generate": "generate",

            # 검색 실패 시 안내 메시지를 반환합니다.
            "fallback": "fallback",
        },
    )

    # 답변 생성이 끝나면 그래프를 종료합니다.
    graph.add_edge("generate", END)

    # fallback 처리가 끝나면 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 그래프를 실행 가능한 형태로 완성합니다.
    return graph.compile()


# 프로그램이 시작될 때 그래프를 한 번 만듭니다.
app = build_graph()


# ==================================================
# 문서 하나에서 출처 정보를 가져오는 함수
# ==================================================
def get_source(document):
    """
    문서의 메타데이터에서 파일명과 페이지 번호를 가져옵니다.
    """

    # 문서의 메타데이터를 가져옵니다.
    metadata = document.metadata

    # 파일명 후보를 순서대로 확인합니다.
    filename = metadata.get("filename")

    if not filename:
        filename = metadata.get("file_name")

    if not filename:
        filename = metadata.get("source")

    if not filename:
        filename = "unknown"

    # 페이지 번호 후보를 순서대로 확인합니다.
    page = metadata.get("page_no")

    if not page:
        page = metadata.get("page")

    if not page:
        page = metadata.get("pageno")

    if not page:
        page = "?"

    # 파일명과 페이지 번호를 하나의 딕셔너리로 반환합니다.
    return {
        "file": filename,
        "page": page,
    }


# ==================================================
# 외부에서 사용하는 대표 함수
# ==================================================
def ask(question):
    """
    질문을 그래프에 전달하고 최종 답변을 반환합니다.
    """

    # 질문을 그래프가 사용할 초기 상태로 만듭니다.
    initial_state = make_initial_state(question)

    # 초기 상태를 그래프에 전달하여 실행합니다.
    final = app.invoke(
        initial_state,
        {
            # 그래프가 너무 오래 반복되지 않도록 제한합니다.
            "recursion_limit": 20
        },
    )

    # 출처를 저장할 빈 리스트를 만듭니다.
    sources = []

    # 검색된 문서를 하나씩 확인합니다.
    for document in final.get("documents", []):
        # 문서에서 파일명과 페이지 정보를 가져옵니다.
        source = get_source(document)

        # 출처 목록에 추가합니다.
        sources.append(source)

    # 외부 프로그램에서 사용할 결과를 반환합니다.
    return {
        # 최종 답변입니다.
        "answer": final["answer"],

        # 답변 생성에 사용된 문서 목록입니다.
        "sources": sources,

        # 답변에 인용이 포함되었는지 나타냅니다.
        "cited": final.get("has_citation", False),

        # 그래프가 실행된 과정을 기록합니다.
        "log": final.get("log", []),

        # 함수가 정상적으로 실행되었음을 나타냅니다.
        "ok": True,
    }


# ==================================================
# 이 파일을 직접 실행했을 때의 동작
# ==================================================
if __name__ == "__main__":
    # 그래프의 노드와 연결 구조를 출력합니다.
    print(app.get_graph().draw_ascii())