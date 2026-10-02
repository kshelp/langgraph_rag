# 이 파일은 LangGraph가 실제로 어떤 경로로 실행되는지 확인합니다.
# 질문을 입력하면 retrieve, generate, fallback 노드의 실행 순서를 화면에 출력합니다.

from graph import app
from graph_state2 import make_initial_state
import warnings
warnings.filterwarnings("ignore")


# ==================================================
# 실행 경로를 확인할 질문 목록
# ==================================================
QUESTIONS = [
    # 검색 결과가 있을 것으로 예상되는 질문입니다.
    "환불은 며칠 이내인가요?",

    # 검색 결과가 없을 것으로 예상되는 질문입니다.
    "대표이사가 누구인가요?",
]


# ==================================================
# 하나의 질문에 대해 그래프 실행 과정을 출력하는 함수
# ==================================================
def show_graph_process(question):
    """
    질문 하나를 그래프에 전달하고 실행 과정을 출력합니다.
    """

    # 질문 제목을 출력합니다.
    print()
    print("=" * 55)
    print("Q:", question)
    print("=" * 55)

    # 질문을 그래프가 사용할 초기 상태로 만듭니다.
    initial_state = make_initial_state(question)

    # 그래프를 실행하면서 각 노드의 결과를 하나씩 받습니다.
    for step in app.stream(initial_state):

        # 현재 실행된 노드와 상태 변경 내용을 확인합니다.
        for node, update in step.items():

            # 어떤 노드가 실행되었는지 출력합니다.
            print(f"  [{node}] {list(update.keys())}")

            # 검색 노드가 실행된 경우 검색 정보를 출력합니다.
            if node == "retrieve":
                print("    검색 점수:", update.get("scores"))
                print(
                    "    검색 성공 여부:",
                    update.get("retrieval_ok"),
                )


# ==================================================
# 모든 질문에 대해 그래프 실행 과정을 확인합니다.
# ==================================================
def main():
    """
    질문 목록을 차례대로 실행합니다.
    """

    # 질문을 하나씩 꺼냅니다.
    for question in QUESTIONS:

        # 현재 질문의 그래프 실행 과정을 출력합니다.
        show_graph_process(question)


# ==================================================
# 이 파일을 직접 실행했을 때 시작합니다.
# ==================================================
if __name__ == "__main__":
    main()