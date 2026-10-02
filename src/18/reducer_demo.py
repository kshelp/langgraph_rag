# ============================================================
# LangGraph에서 Reducer가 있을 때와 없을 때의 차이를 비교합니다.
# Reducer가 없으면 log가 덮어써지고, 있으면 log가 계속 누적됩니다.
# ============================================================

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Reducer가 없는 State
# ============================================================

class StateA(TypedDict):
    """log에 새 값이 들어오면 기존 값을 덮어씁니다."""

    log: list[str]


# ============================================================
# 2. Reducer가 있는 State
# ============================================================

class StateB(TypedDict):
    """log에 새 값이 들어오면 기존 리스트 뒤에 추가합니다."""

    log: Annotated[list[str], operator.add]


# ============================================================
# 3. Node 정의
# ============================================================

def step1(state):
    """1단계 실행 기록을 반환합니다."""

    return {"log": ["1단계 완료"]}


def step2(state):
    """2단계 실행 기록을 반환합니다."""

    return {"log": ["2단계 완료"]}


def step3(state):
    """3단계 실행 기록을 반환합니다."""

    return {"log": ["3단계 완료"]}


# ============================================================
# 4. 실험 A - Reducer 없음
# ============================================================

# StateA를 사용하는 그래프를 만듭니다.
graph_a = StateGraph(StateA)

# 노드를 등록합니다.
graph_a.add_node("step1", step1)
graph_a.add_node("step2", step2)
graph_a.add_node("step3", step3)

# START → step1 → step2 → step3 → END 순서로 연결합니다.
graph_a.add_edge(START, "step1")
graph_a.add_edge("step1", "step2")
graph_a.add_edge("step2", "step3")
graph_a.add_edge("step3", END)

# 실행 가능한 그래프로 만듭니다.
app_a = graph_a.compile()

# 빈 log에서 실행을 시작합니다.
result_a = app_a.invoke({"log": []})

# 최종 log를 출력합니다.
print("Reducer 없음:", result_a["log"])


# ============================================================
# 5. 실험 B - Reducer 있음
# ============================================================

# StateB를 사용하는 그래프를 만듭니다.
graph_b = StateGraph(StateB)

# 노드를 등록합니다.
graph_b.add_node("step1", step1)
graph_b.add_node("step2", step2)
graph_b.add_node("step3", step3)

# START → step1 → step2 → step3 → END 순서로 연결합니다.
graph_b.add_edge(START, "step1")
graph_b.add_edge("step1", "step2")
graph_b.add_edge("step2", "step3")
graph_b.add_edge("step3", END)

# 실행 가능한 그래프로 만듭니다.
app_b = graph_b.compile()

# 빈 log에서 실행을 시작합니다.
result_b = app_b.invoke({"log": []})

# 최종 log를 출력합니다.
print("Reducer 있음:", result_b["log"])