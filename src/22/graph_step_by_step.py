# 이 파일은 RAG 그래프를 세 단계로 나누어 만들어 보는 실습 파일입니다.
# 검색 노드만 연결한 뒤, 답변 생성과 fallback을 차례대로 추가하며 각 단계를 확인합니다.

import os
import sys
import warnings
warnings.filterwarnings("ignore")

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

# LangGraph 그래프 구성 도구입니다.
from langgraph.graph import StateGraph, START, END

# 그래프가 사용할 상태 형식입니다.
from graph_state import RAGState

# 질문을 초기 상태로 만드는 함수입니다.
from graph_state2 import make_initial_state

# 각 작업을 수행하는 노드 함수입니다.
from retrieve import retrieve_node
from generate import generate_node
from fallback import fallback_node


# ==================================================
# 검색 결과에 따라 다음 노드를 결정하는 함수
# ==================================================
def route_after_retrieve(state):
    """
    검색에 성공하면 generate로 이동하고,
    검색에 실패하면 fallback으로 이동합니다.
    """

    # 검색 성공 여부를 확인합니다.
    if state.get("retrieval_ok"):
        return "generate"

    # 검색 실패 시 fallback으로 이동합니다.
    return "fallback"


# ==================================================
# 1단계: retrieve 노드만 연결
# ==================================================
def build_step1_graph():
    """
    검색 노드만 연결한 가장 간단한 그래프를 만듭니다.
    """

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # 검색 노드를 등록합니다.
    graph.add_node("retrieve", retrieve_node)

    # START → retrieve → END로 연결합니다.
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)

    # 실행 가능한 그래프로 완성합니다.
    return graph.compile()


# ==================================================
# 1단계 그래프 실행
# ==================================================
def test_step1():
    """
    검색 노드가 문서를 제대로 찾는지 확인합니다.
    """

    print()
    print("=" * 60)
    print("1단계: retrieve 노드만 연결")
    print("=" * 60)

    # 1단계 그래프를 만듭니다.
    app = build_step1_graph()

    # 질문을 초기 상태로 만들어 그래프에 전달합니다.
    result = app.invoke(
        make_initial_state("환불은 며칠 이내인가요?")
    )

    # 검색된 문서 개수를 출력합니다.
    print("검색 결과:", len(result.get("documents", [])), "건")

    # 검색 점수를 출력합니다.
    print("점수:", result.get("scores", []))

    # 검색 성공 여부를 출력합니다.
    print("검색 성공 여부:", result.get("retrieval_ok"))


# ==================================================
# 2단계: retrieve와 generate 연결
# ==================================================
def build_step2_graph():
    """
    검색 후 답변을 생성하는 그래프를 만듭니다.
    """

    # RAGState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(RAGState)

    # 검색 노드를 등록합니다.
    graph.add_node("retrieve", retrieve_node)

    # 답변 생성 노드를 등록합니다.
    graph.add_node("generate", generate_node)

    # START → retrieve → generate → END로 연결합니다.
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # 실행 가능한 그래프로 완성합니다.
    return graph.compile()


# ==================================================
# 2단계 그래프 실행
# ==================================================
def test_step2():
    """
    검색 결과를 이용해 답변을 생성하는지 확인합니다.
    """

    print()
    print("=" * 60)
    print("2단계: generate 노드 추가")
    print("=" * 60)

    # 2단계 그래프를 만듭니다.
    app = build_step2_graph()

    # 질문을 초기 상태로 만들어 그래프에 전달합니다.
    result = app.invoke(
        make_initial_state("환불은 며칠 이내인가요?")
    )

    # 생성된 답변을 출력합니다.
    print("답변:")
    print(result.get("answer", ""))


# ==================================================
# 3단계: 조건 분기와 fallback 추가
# ==================================================
def build_step3_graph():
    """
    검색 성공과 검색 실패를 나누는 최종 그래프를 만듭니다.
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

    # 검색 결과에 따라 다음 노드를 결정합니다.
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            # 검색 성공 시 답변을 생성합니다.
            "generate": "generate",

            # 검색 실패 시 fallback을 실행합니다.
            "fallback": "fallback",
        },
    )

    # 답변 생성 후 그래프를 종료합니다.
    graph.add_edge("generate", END)

    # fallback 처리 후 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 실행 가능한 그래프로 완성합니다.
    return graph.compile()


# ==================================================
# 3단계 그래프 실행
# ==================================================
def test_step3():
    """
    정상 질문과 검색 실패 질문의 경로를 확인합니다.
    """

    print()
    print("=" * 60)
    print("3단계: 조건 분기와 fallback 추가")
    print("=" * 60)

    # 3단계 그래프를 만듭니다.
    app = build_step3_graph()

    # ----------------------------------------------
    # 정상 질문 테스트
    # ----------------------------------------------
    print()
    print("[정상 질문]")

    # 정상적인 질문을 그래프에 전달합니다.
    result1 = app.invoke(
        make_initial_state("환불은 며칠 이내인가요?")
    )

    # 생성된 답변을 출력합니다.
    print("답변:")
    print(result1.get("answer", ""))

    # 실행 과정을 출력합니다.
    print("실행 로그:", result1.get("log", []))

    # ----------------------------------------------
    # 검색 실패 질문 테스트
    # ----------------------------------------------
    print()
    print("[검색 실패 질문]")

    # 문서에 없는 질문을 그래프에 전달합니다.
    result2 = app.invoke(
        make_initial_state("대표이사가 누구인가요?")
    )

    # fallback 답변을 출력합니다.
    print("답변:")
    print(result2.get("answer", ""))

    # 실행 과정을 출력합니다.
    print("실행 로그:", result2.get("log", []))


# ==================================================
# 세 단계 그래프의 구조 출력
# ==================================================
def print_graphs():
    """
    각 단계의 그래프 구조를 화면에 출력합니다.
    """

    print()
    print("=" * 60)
    print("그래프 구조 확인")
    print("=" * 60)

    # 1단계 그래프 구조를 출력합니다.
    print()
    print("[1단계 그래프]")
    app1 = build_step1_graph()
    print(app1.get_graph().draw_ascii())

    # 2단계 그래프 구조를 출력합니다.
    print()
    print("[2단계 그래프]")
    app2 = build_step2_graph()
    print(app2.get_graph().draw_ascii())

    # 3단계 그래프 구조를 출력합니다.
    print()
    print("[3단계 그래프]")
    app3 = build_step3_graph()
    print(app3.get_graph().draw_ascii())


# ==================================================
# 프로그램 시작
# ==================================================
if __name__ == "__main__":

    # 1단계 그래프를 실행합니다.
    test_step1()

    # 2단계 그래프를 실행합니다.
    test_step2()

    # 3단계 그래프를 실행합니다.
    test_step3()

    # 세 그래프의 구조를 출력합니다.
    print_graphs()