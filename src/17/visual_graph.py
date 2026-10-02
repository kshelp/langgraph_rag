from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

import io
import matplotlib.pyplot as plt
from PIL import Image

class SimpleRAGState(TypedDict):
    query: str      # 사용자의 질문
    results: list   # 검색 결과 목록
    answer: str     # 최종 답변

def search_node(state: SimpleRAGState):
    query = state["query"]
    # 현재 State에서 질문을 읽습니다.
    print("[검색 노드] 질문:", state["query"])

    # 실제 검색 대신 고정된 문서 목록을 준비합니다.
    if "가격" in query:
        results = [
            "문서1: 제품 가격은 10,000원입니다.",
            "문서2: 배송비는 무료입니다.",
            "문서3: 현재 할인 행사가 진행 중입니다."
        ]
    else:
        results = []

    print("[검색] 결과:", results)

    # LangGraph가 results 항목을 갱신하도록 반환합니다.
    return {"results": results}

def answer_node(state: SimpleRAGState):
    # 앞의 검색 노드가 반환한 결과를 읽습니다.
    print("[답변 노드] 검색 결과:", state["results"])

    # 실제 LLM 대신 고정된 답변을 준비합니다.
    answer = "검색 결과를 바탕으로 만든 답변입니다."

    # LangGraph가 answer 항목을 갱신하도록 반환합니다.
    return {"answer": answer}

def check_search_quality(state: SimpleRAGState):
    # 검색 결과가 하나 이상이면 답변 경로를 선택합니다.
    if len(state["results"]) > 0:
        print("[판단] 검색 성공 → 답변")
        return "good"

    # 검색 결과가 없으면 재검색 경로를 선택합니다.
    print("[판단] 검색 실패 → 다시 검색")
    return "bad"


graph = StateGraph(SimpleRAGState)

graph.add_node("search", search_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "search")

graph.add_conditional_edges(
    "search",
    check_search_quality,
    {
        "good": "answer",  # 성공하면 답변 노드로 이동
        "bad": "search"    # 실패하면 검색 노드로 돌아감
    }
)

graph.add_edge("answer", END)

app = graph.compile()


print("Graph 이미지를 생성합니다...")

image_data = app.get_graph().draw_mermaid_png()

# PNG 바이트 데이터를 메모리상의 파일처럼 다룹니다.
image_file = io.BytesIO(image_data)

# 메모리상의 PNG 데이터를 PIL 이미지 객체로 엽니다.
image = Image.open(image_file)


# 8. 그래프 이미지를 화면에 표시합니다.
plt.figure(figsize=(8, 6))  # 그림의 가로·세로 크기를 지정합니다.
plt.imshow(image)          # PIL 이미지를 표시합니다.
plt.axis("off")            # 좌표축과 눈금을 숨깁니다.
plt.title("LangGraph - Conditional RAG Graph")
plt.show()                 # 그림 창을 화면에 띄웁니다.


