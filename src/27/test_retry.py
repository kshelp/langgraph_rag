# 이 파일은 27차시 RAG 그래프의 실행 과정을 확인합니다.
# 각 노드가 어떤 순서로 실행되었는지와 로그 내용을 출력합니다.


import os
import sys


# ============================================================
# 필요한 모듈 경로 등록
# ============================================================

# 현재 파일이 있는 폴더의 경로입니다.
CURRENT_DIR = os.path.dirname(__file__)

# 18차시의 State 모듈을 Python 검색 경로에 추가합니다.
STATE_DIR = os.path.join(CURRENT_DIR, "..", "18")
sys.path.insert(0, STATE_DIR)


# ============================================================
# 그래프와 초기 State 함수 가져오기
# ============================================================

# 27차시에서 완성한 LangGraph를 가져옵니다.
from graph27 import app

# 질문을 초기 State로 변환하는 함수를 가져옵니다.
from graph_state2 import make_initial_state


# ============================================================
# 그래프 실행 및 로그 출력
# ============================================================

def trace_graph(question):
    """질문을 그래프에 전달하고 노드 실행 과정을 출력합니다."""

    # 질문을 LangGraph가 사용할 초기 State로 변환합니다.
    initial_state = make_initial_state(question)

    # 그래프를 실행하면서 각 노드의 결과를 순서대로 받습니다.
    steps = app.stream(
        initial_state,
        {
            "recursion_limit": 25
        },
    )

    # 실행된 노드와 각 노드의 결과를 하나씩 확인합니다.
    for step in steps:

        # step에는 현재 실행된 노드 이름과 결과가 들어 있습니다.
        for node_name, update in step.items():

            # 현재 실행된 노드 이름을 출력합니다.
            print()
            print(f"[{node_name}]")

            # 해당 노드가 남긴 로그를 출력합니다.
            logs = update.get("log", [])

            for log in logs:
                print(f"    {log}")


# ============================================================
# 프로그램 시작점
# ============================================================

if __name__ == "__main__":

    # 재시도 상황을 확인하기 위한 질문입니다.
    question = "환불 방법과 대표이사를 알려주세요"

    # 질문을 그래프에 전달하고 전체 실행 과정을 출력합니다.
    trace_graph(question)