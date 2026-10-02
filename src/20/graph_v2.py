# ============================================================
# LangGraph 기반 기본 RAG 그래프 구성
# 검색 결과가 있으면 답변을 생성하고, 없으면 안내 메시지를 출력합니다.
# 완성된 그래프 구조를 화면에 출력하고 graph.png 파일로 저장합니다.
# ============================================================

import os
import sys

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 필요한 모듈 경로 설정
# ============================================================

# 현재 파일이 있는 폴더를 가져옵니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 18차시 폴더를 Python 모듈 검색 경로에 추가합니다.
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "18"))

# 19차시 폴더를 Python 모듈 검색 경로에 추가합니다.
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "19"))


# ============================================================
# 2. RAG 그래프에 필요한 모듈 가져오기
# ============================================================

# 그래프에서 사용할 State 자료형입니다.
from graph_state import RAGState

# 문서를 검색하는 노드입니다.
from retrieve import retrieve_node

# 검색된 문서를 이용해 답변을 생성하는 노드입니다.
from generate import generate_node

# 검색 결과가 없을 때 안내 메시지를 만드는 노드입니다.
from fallback import fallback_node

# 검색 성공 여부에 따라 다음 경로를 결정하는 함수입니다.
from routes import route_after_retrieve


# ============================================================
# 3. LangGraph 그래프 생성
# ============================================================

# RAGState를 사용하는 빈 그래프를 만듭니다.
graph = StateGraph(RAGState)


# ============================================================
# 4. Node 등록
# ============================================================

# 문서 검색 노드를 등록합니다.
graph.add_node("retrieve", retrieve_node)

# 답변 생성 노드를 등록합니다.
graph.add_node("generate", generate_node)

# 검색 실패 안내 노드를 등록합니다.
graph.add_node("fallback", fallback_node)


# ============================================================
# 5. Edge 연결
# ============================================================

# 그래프가 시작되면 먼저 문서를 검색합니다.
graph.add_edge(START, "retrieve")


# 검색 결과에 따라 다음 노드를 선택합니다.
graph.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {
        "ok": "generate",
        "empty": "fallback"
    }
)


# 답변 생성이 끝나면 그래프를 종료합니다.
graph.add_edge("generate", END)

# 검색 실패 안내가 끝나도 그래프를 종료합니다.
graph.add_edge("fallback", END)


# ============================================================
# 6. 그래프 실행 준비
# ============================================================

# 작성한 그래프를 실제 실행 가능한 형태로 만듭니다.
app = graph.compile()


# ============================================================
# 7. 그래프 구조 확인
# ============================================================

# 그래프 구조를 문자 형태로 화면에 출력합니다.
print(app.get_graph().draw_ascii())


# ============================================================
# 8. 그래프를 PNG 이미지로 저장
# ============================================================

# 그래프 구조를 PNG 이미지 데이터로 변환합니다.
png_data = app.get_graph().draw_mermaid_png()

# PNG 데이터를 graph.png 파일로 저장합니다.
with open("graph.png", "wb") as file:
    file.write(png_data)

print("graph.png 저장 완료")