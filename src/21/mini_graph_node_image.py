# 이 코드는 더미 데이터로 Modular RAG 그래프를 구성합니다.
# 검색, 답변 생성, 검증, 재시도, fallback 흐름을 확인합니다.

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. State 정의
# ============================================================

class MiniState(TypedDict):
    # 사용자가 입력한 질문
    question: str

    # 검색된 문서 목록
    documents: list[str]

    # 생성된 답변
    answer: str

    # 검증 결과
    # pass: 통과, retry: 재시도, giveup: 포기
    grade: str

    # 재시도 횟수
    retries: int


# ============================================================
# 초기 State 생성
# ============================================================

def init(question: str) -> MiniState:
    """
    그래프 실행에 사용할 초기 상태를 만듭니다.
    """

    # 모든 값은 기본값으로 시작합니다.
    return {
        "question": question,
        "documents": [],
        "answer": "",
        "grade": "",
        "retries": 0,
    }


# ============================================================
# 2. 노드 정의
# ============================================================

def retrieve(state: MiniState) -> dict:
    """
    질문에 대한 검색 결과를 만듭니다.

    질문에 '없는'이라는 단어가 있으면
    검색 결과가 없는 상황을 흉내 냅니다.
    """

    # 검색 결과를 저장할 리스트입니다.
    documents = []

    # 질문에 '없는'이 포함되어 있는지 확인합니다.
    if "없는" in state["question"]:
        # 검색 실패 상황입니다.
        documents = []
    else:
        # 검색 성공 상황을 흉내 냅니다.
        documents = ["조각A", "조각B", "조각C"]

    # 검색 결과의 개수를 출력합니다.
    print(f"→ 검색: {len(documents)}건")

    # 변경된 documents만 반환합니다.
    return {
        "documents": documents
    }


def generate(state: MiniState) -> dict:
    """
    검색 결과를 바탕으로 답변을 만듭니다.

    첫 번째 답변은 부실하게 만들고,
    재시도 후에는 좋은 답변을 만듭니다.
    """

    # 현재 재시도 횟수를 확인합니다.
    retry_count = state["retries"]

    # 답변의 품질을 저장할 변수입니다.
    quality = ""

    # 첫 번째 답변인지 확인합니다.
    if retry_count == 0:
        # 첫 번째 답변은 부실한 답변으로 처리합니다.
        quality = "부실한"
    else:
        # 재시도 후에는 좋은 답변으로 처리합니다.
        quality = "좋은"

    # 검색된 문서 개수를 확인합니다.
    document_count = len(state["documents"])

    # 가짜 답변을 만듭니다.
    answer = f"{document_count}건 근거로 만든 {quality} 답변"

    # 생성된 답변을 출력합니다.
    print(f"→ 생성: {answer}")

    # 변경된 answer만 반환합니다.
    return {
        "answer": answer
    }


def verify(state: MiniState) -> dict:
    """
    생성된 답변을 검증합니다.

    답변에 '좋은'이라는 단어가 있으면 통과시키고,
    그렇지 않으면 재시도 대상으로 처리합니다.
    """

    # 현재 답변에 '좋은'이라는 단어가 있는지 확인합니다.
    if "좋은" in state["answer"]:
        # 검증을 통과시킵니다.
        print("→ 검증: 통과")

        return {
            "grade": "pass"
        }

    # 검증에 실패하면 재시도 상태를 반환합니다.
    print("→ 검증: 실패")

    return {
        "grade": "retry"
    }


def bump(state: MiniState) -> dict:
    """
    재시도 횟수를 1 증가시킵니다.
    """

    # 현재 재시도 횟수에 1을 더합니다.
    retry_count = state["retries"] + 1

    # 재시도 횟수를 출력합니다.
    print(f"→ 재시도: {retry_count}회")

    # 변경된 retries만 반환합니다.
    return {
        "retries": retry_count
    }


def fallback(state: MiniState) -> dict:
    """
    검색에 실패했거나 재시도 횟수를 초과했을 때
    안내 답변을 반환합니다.
    """

    # fallback 노드가 실행되었음을 출력합니다.
    print("→ fallback")

    # 안내 답변과 포기 상태를 반환합니다.
    return {
        "answer": "자료를 찾지 못했습니다.",
        "grade": "giveup",
    }


# ============================================================
# 3. 분기 함수 정의
# ============================================================

# 허용할 최대 재시도 횟수입니다.
MAX_RETRY = 2


def route_retrieve(state: MiniState) -> str:
    """
    검색 결과가 있는지 확인하고 다음 경로를 결정합니다.
    """

    # documents에 문서가 있으면 정상 경로로 이동합니다.
    if state["documents"]:
        return "ok"

    # documents가 비어 있으면 fallback 경로로 이동합니다.
    return "empty"


def route_verify(state: MiniState) -> str:
    """
    검증 결과와 재시도 횟수를 확인하고
    다음 실행 경로를 결정합니다.
    """

    # 답변 검증에 성공하면 그래프를 종료합니다.
    if state["grade"] == "pass":
        return "done"

    # 최대 재시도 횟수에 도달하면 포기합니다.
    if state["retries"] >= MAX_RETRY:
        return "giveup"

    # 아직 재시도할 수 있으면 bump로 이동합니다.
    return "retry"


# ============================================================
# 4. 그래프 만들기
# ============================================================

def make_graph():
    """
    노드와 엣지를 연결하여 실행 가능한 그래프를 만듭니다.
    """

    # MiniState를 사용하는 그래프를 만듭니다.
    graph = StateGraph(MiniState)

    # 노드를 그래프에 등록합니다.
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("bump", bump)
    graph.add_node("fallback", fallback)

    # 그래프 시작점에서 retrieve로 이동합니다.
    graph.add_edge(START, "retrieve")

    # 검색 결과에 따라 다음 노드를 결정합니다.
    graph.add_conditional_edges(
        "retrieve",
        route_retrieve,
        {
            "ok": "generate",
            "empty": "fallback",
        },
    )

    # 답변 생성 후 검증을 실행합니다.
    graph.add_edge("generate", "verify")

    # 검증 결과에 따라 다음 노드를 결정합니다.
    graph.add_conditional_edges(
        "verify",
        route_verify,
        {
            "done": END,
            "retry": "bump",
            "giveup": "fallback",
        },
    )

    # 재시도 횟수를 증가시킨 뒤 다시 답변을 생성합니다.
    graph.add_edge("bump", "generate")

    # fallback 이후 그래프를 종료합니다.
    graph.add_edge("fallback", END)

    # 그래프를 컴파일하여 실행 가능한 객체로 만듭니다.
    return graph.compile()


# ============================================================
# 5. 그래프 실행 및 이미지 저장
# ============================================================

if __name__ == "__main__":

    # 그래프를 만듭니다.
    app = make_graph()

    print("■ 그래프 실행")

    # 그래프를 단계별로 실행합니다.
    for step in app.stream(init("환불 규정은?")):
        print(step)

    # 그래프 구조를 PNG 이미지로 변환합니다.
    image_data = app.get_graph().draw_mermaid_png()

    # 이미지 파일로 저장합니다.
    with open("graph_structure.png", "wb") as file:
        file.write(image_data)

    print("graph_structure.png 저장 완료")