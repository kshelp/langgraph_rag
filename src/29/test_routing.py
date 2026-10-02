# ============================================================
# Classifier Test
# ============================================================
# 여러 질문을 그래프에 전달하여 분류 결과를 확인합니다.
# 실제 분류 결과와 예상 분류 결과를 비교하여
# 분류 노드가 정상적으로 작동하는지 테스트합니다.
# ============================================================


import os
import sys


# ============================================================
# 모듈 경로 설정
# ============================================================

# 현재 테스트 파일이 있는 폴더입니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 현재 폴더의 상위 src 폴더를 찾습니다.
SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

# src 폴더에서 graph29와 State 모듈을 찾을 수 있도록 추가합니다.
sys.path.insert(0, SRC_DIR)


# ============================================================
# 그래프와 초기 State 가져오기
# ============================================================

# 분류 그래프를 가져옵니다.
from graph29 import app

# 질문을 초기 State로 만들어 주는 함수를 가져옵니다.
from graph_state2 import make_initial_state


# ============================================================
# 테스트 질문과 예상 결과
# ============================================================

# 각 질문과 기대하는 분류 결과를 저장합니다.
TEST_CASES = [
    ("안녕하세요", "greeting"),
    ("고맙습니다", "greeting"),
    ("125 + 340 * 2", "calc"),
    ("오늘 날씨 어때요?", "scope"),
    ("환불은 며칠 이내인가요?", "document"),
    ("휴가 신청은 어떻게 하나요?", "document"),
]


# ============================================================
# 테스트 실행 함수
# ============================================================

def run_tests():
    """
    테스트 질문을 그래프에 전달하고 결과를 출력합니다.
    """

    # 테스트 질문을 하나씩 확인합니다.
    for question, expected_intent in TEST_CASES:

        # 질문을 초기 State로 변환합니다.
        initial_state = make_initial_state(question)

        # 그래프를 실행합니다.
        result = app.invoke(
            initial_state,
            {
                "recursion_limit": 25,
            },
        )

        # 실제 분류 결과를 가져옵니다.
        actual_intent = result.get("intent", "")

        # 실제 결과와 예상 결과를 비교합니다.
        if actual_intent == expected_intent:
            mark = "✓"
        else:
            mark = "✗"

        # 답변이 없어도 오류가 발생하지 않도록 기본값을 사용합니다.
        answer = result.get("answer", "")

        # 질문별 테스트 결과만 출력합니다.
        print()
        print(f"{mark} 질문: {question}")
        print(f"   실제 분류: {actual_intent}")
        print(f"   예상 분류: {expected_intent}")
        print(f"   답변: {answer[:60]}")


if __name__ == "__main__":
    run_tests()