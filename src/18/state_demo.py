from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class DemoState(TypedDict):
    """그래프 전체에서 공유할 데이터를 정의합니다."""
    question: str          # 사용자 질문
    documents: list[str]   # 검색된 문서
    answer: str            # 최종 답변

def retrieve(state: DemoState):
    """질문을 받아 관련 문서를 검색하는 노드입니다."""

    # 현재 State가 어떻게 전달되었는지 확인합니다.
    print("[retrieve] 받은 State:", state)

    # 실제 검색 대신 테스트용 문서를 반환합니다.
    documents = ["조각A", "조각B"]

    # documents만 State에 업데이트합니다.
    return {"documents": documents}


def generate(state: DemoState):
    """검색된 문서를 이용해 답변을 만드는 노드입니다."""

    # retrieve에서 추가한 documents도 함께 전달됩니다.
    print("[generate] 받은 State:", state)

    # 검색된 문서 개수를 계산합니다.
    document_count = len(state["documents"])

    # 문서 개수를 이용해 간단한 답변을 만듭니다.
    answer = f"{document_count}건의 근거로 만든 답변"

    # answer만 State에 업데이트합니다.
    return {"answer": answer}

# DemoState를 사용하는 그래프를 만듭니다.
graph = StateGraph(DemoState)

# 그래프에 노드를 등록합니다.
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

# START → retrieve → generate → END 순서로 연결합니다.
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# 실행 가능한 그래프로 완성합니다.
app = graph.compile()

# 처음 사용할 State를 만듭니다.
initial_state = {
    "question": "환불 규정은?",
    "documents": [],
    "answer": ""
}

# 그래프를 실행합니다.
final_state = app.invoke(initial_state)

# 모든 노드 실행이 끝난 최종 State를 출력합니다.
print("\n[최종] State:", final_state)