# ============================================================
# LangGraph를 이용해 기본 RAG 그래프를 구성하고 실행합니다.
# 검색 성공 시 답변을 생성하고, 검색 실패 시 안내 메시지를 출력합니다.
# 두 가지 질문으로 정상 검색과 검색 실패 흐름을 테스트합니다.
# ============================================================

import os
import sys
import warnings

# 불필요한 경고 메시지를 숨깁니다.
warnings.filterwarnings("ignore")


# ============================================================
# 1. 모듈 경로 설정
# ============================================================

# 현재 파일이 있는 폴더를 가져옵니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 필요한 차시 폴더를 Python 검색 경로에 추가합니다.
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "18"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "19"))


# ============================================================
# 2. 필요한 모듈 가져오기
# ============================================================

from langgraph.graph import StateGraph, START, END

# 그래프에서 사용할 State입니다.
from graph_state import RAGState

# 문서를 검색하는 노드입니다.
from retrieve import retrieve_node

# 검색된 문서로 답변을 만드는 노드입니다.
from generate import generate_node

# 검색 결과가 없을 때 안내하는 노드입니다.
from fallback import fallback_node

# 검색 결과에 따라 다음 경로를 결정합니다.
from routes import route_after_retrieve


# ============================================================
# 3. 그래프 생성
# ============================================================

# RAGState를 사용하는 빈 그래프를 만듭니다.
graph = StateGraph(RAGState)


# ============================================================
# 4. 노드 등록
# ============================================================

# 문서 검색 노드를 등록합니다.
graph.add_node("retrieve", retrieve_node)

# 답변 생성 노드를 등록합니다.
graph.add_node("generate", generate_node)

# 검색 실패 안내 노드를 등록합니다.
graph.add_node("fallback", fallback_node)


# ============================================================
# 5. 노드 연결
# ============================================================

# 그래프가 시작되면 retrieve 노드로 이동합니다.
graph.add_edge(START, "retrieve")

# 검색 결과에 따라 generate 또는 fallback으로 이동합니다.
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

# 검색 실패 안내가 끝나면 그래프를 종료합니다.
graph.add_edge("fallback", END)


# ============================================================
# 6. 그래프 컴파일
# ============================================================

# 작성한 그래프를 실제 실행 가능한 형태로 만듭니다.
app = graph.compile()


# ============================================================
# 7. 그래프 실행 함수
# ============================================================

def run(question):
    """질문을 그래프에 전달하고 최종 결과를 출력합니다."""

    # 그래프 실행에 필요한 초기 State를 만듭니다.
    state = {
        "question": question,
        "query": question,
        "retries": 0,
        "log": [],
        "tried_queries": [],
    }

    # 초기 State를 전달하여 그래프를 실행합니다.
    result = app.invoke(state)

    # 질문과 답변을 출력합니다.
    print("Q:", question)
    print("A:", result["answer"][:70])

    # 각 노드에서 기록한 실행 과정을 출력합니다.
    for line in result["log"]:
        print("   ", line)

    print("-" * 55)


# ============================================================
# 8. 테스트 실행
# ============================================================

# 검색 결과가 있는 질문을 테스트합니다.
run("환불은 며칠 이내인가요?")

# 검색 결과가 없는 질문을 테스트합니다.
run("대표이사 이름은?")