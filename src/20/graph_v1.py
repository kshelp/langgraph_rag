# ============================================================
# RAG의 전체 실행 흐름을 LangGraph로 구성합니다.
# 검색 성공 시 답변을 생성하고, 검색 실패 시 안내 메시지를 출력합니다.
# ============================================================

import os
import sys

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 다른 폴더에 있는 모듈을 사용할 수 있도록 경로를 추가합니다.
# ============================================================

CURRENT_DIR = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "18"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "19"))


# ============================================================
# 2. 그래프 구성에 필요한 기능을 가져옵니다.
# ============================================================

# 그래프에서 사용할 State
from graph_state import RAGState

# 검색을 담당하는 Node
from retrieve import retrieve_node

# 답변 생성을 담당하는 Node
from generate import generate_node

# 검색 실패 시 안내를 담당하는 Node
from fallback import fallback_node

# 검색 결과에 따라 다음 경로를 결정하는 함수
from routes import route_after_retrieve


# ============================================================
# 3. RAGState를 사용하는 그래프를 만듭니다.
# ============================================================

graph = StateGraph(RAGState)


# ============================================================
# 4. 그래프에서 사용할 Node를 등록합니다.
# ============================================================

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("fallback", fallback_node)


# ============================================================
# 5. 그래프의 실행 순서를 연결합니다.
# ============================================================

# 그래프가 시작되면 먼저 문서를 검색합니다.
graph.add_edge(START, "retrieve")


# 검색 결과에 따라 다음 Node를 선택합니다.
graph.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {
        "ok": "generate",       # 검색 성공 → 답변 생성
        "empty": "fallback"     # 검색 실패 → 안내 메시지
    }
)


# 답변 생성이 끝나면 그래프를 종료합니다.
graph.add_edge("generate", END)

# 안내 메시지 출력이 끝나면 그래프를 종료합니다.
graph.add_edge("fallback", END)


# ============================================================
# 6. 완성된 그래프를 실행 가능한 형태로 만듭니다.
# ============================================================

app = graph.compile()
