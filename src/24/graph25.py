# ============================================================
# Graph 24·25
# Retriever Node와 Generator Node를 LangGraph에 연결합니다.
# 검색 성공 여부에 따라 답변 생성 또는 안내 메시지로 이동합니다.
# ============================================================

import os
import sys
import warnings


# 현재 파일이 있는 폴더
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# 필요한 차시 폴더를 Python 검색 경로에 추가합니다.
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "18"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "19"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "25"))


# 불필요한 경고 메시지를 숨깁니다.
warnings.filterwarnings("ignore")


# LangGraph의 그래프 구성 도구를 가져옵니다.
from langgraph.graph import StateGraph, START, END

# 그래프 State의 자료형을 가져옵니다.
from graph_state import RAGState

# 질문을 State 형태로 만들어 주는 함수를 가져옵니다.
from graph_state2 import make_initial_state

# 검색 노드를 가져옵니다.
from retriever import retriever_node

# 생성 노드를 가져옵니다.
from generator import generator_node

# 검색 실패 시 사용할 안내 노드를 가져옵니다.
from fallback import fallback_node


def route_after_retrieve(state: RAGState) -> str:
    """
    검색 결과에 따라 다음 노드를 결정합니다.

    검색 성공:
        generate 노드로 이동합니다.

    검색 실패:
        fallback 노드로 이동합니다.
    """

    # State에서 검색 성공 여부를 가져옵니다.
    retrieval_ok = state.get("retrieval_ok")

    # 검색에 성공했으면 "ok"를 반환합니다.
    if retrieval_ok:
        return "ok"

    # 검색에 실패했으면 "empty"를 반환합니다.
    return "empty"


def build_graph():
    """
    Retriever, Generator, Fallback 노드를 연결합니다.

    그래프의 흐름:
        START
          ↓
        retrieve
          ├─ 성공 → generate → END
          └─ 실패 → fallback → END
    """

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # 검색 노드를 등록합니다.
    graph.add_node("retrieve", retriever_node)

    # 생성 노드를 등록합니다.
    graph.add_node("generate", generator_node)

    # 검색 실패 처리 노드를 등록합니다.
    graph.add_node("fallback", fallback_node)

    # 그래프 시작점에서 retrieve로 이동합니다.
    graph.add_edge(START, "retrieve")

    # retrieve 결과에 따라 다음 노드를 선택합니다.
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "ok": "generate",
            "empty": "fallback",
        },
    )

    # 답변 생성이 끝나면 그래프를 종료합니다.
    graph.add_edge("generate", END)

    # fallback 안내가 끝나면 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 그래프를 실행할 수 있는 애플리케이션으로 컴파일합니다.
    application = graph.compile()

    return application


# 그래프를 한 번만 생성합니다.
app = build_graph()


def ask(question: str) -> dict:
    """
    질문 하나를 그래프에 전달하고 최종 결과를 반환합니다.

    입력:
        question: 사용자의 질문

    출력:
        answer: 최종 답변
        sources: 답변에 사용된 문서 출처
        cited: 인용 번호 존재 여부
        log: 전체 처리 기록
        ok: 실행 성공 여부
    """

    # 질문을 초기 State로 변환합니다.
    initial_state = make_initial_state(question)

    # 그래프를 실행합니다.
    final_state = app.invoke(
        initial_state,
        {
            "recursion_limit": 20
        },
    )

    # 최종 State에서 필요한 정보만 골라 반환합니다.
    return {
        "answer": final_state.get("answer", ""),
        "sources": get_sources(final_state),
        "cited": final_state.get("has_citation", False),
        "log": final_state.get("log", []),
        "ok": True,
    }


def get_sources(state: dict) -> list:
    """
    State의 documents에서 파일명과 페이지 정보를 추출합니다.
    """

    # 검색된 문서 목록을 가져옵니다.
    documents = state.get("documents", [])

    # 출처 목록을 저장할 변수입니다.
    sources = []

    # 문서를 하나씩 확인합니다.
    for document in documents:

        # 문서의 메타데이터를 가져옵니다.
        metadata = document.metadata

        # 파일명과 페이지 번호를 출처 목록에 추가합니다.
        sources.append(
            {
                "file": metadata.get("filename", "?"),
                "page": metadata.get("page_no", "?"),
            }
        )

    return sources


# 이 파일을 직접 실행했을 때만 테스트합니다.
if __name__ == "__main__":

    # 테스트할 질문 목록입니다.
    questions = [
        "환불은 며칠 이내인가요?",
        "반품하고 싶은데 언제까지?",
    ]

    # 질문을 하나씩 그래프에 전달합니다.
    for question in questions:

        # 전체 그래프를 실행합니다.
        result = ask(question)

        # 결과를 출력합니다.
        print("\n" + "=" * 55)
        print(f"질문: {question}")
        print(f"답변: {result['answer'][:110]}")
        print(f"출처: {result['sources']}")
        print(f"인용 정상: {result['cited']}")
        print(f"실행 성공: {result['ok']}")
        print(f"처리 기록: {result['log']}")