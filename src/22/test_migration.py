# 이 파일은 15차시의 일반 함수 방식과 22차시의 LangGraph 방식을 비교합니다.
# 같은 질문을 두 방식으로 실행한 뒤, 답변이 같은지 확인하는 회귀 테스트 파일입니다.

import os
import sys
import warnings
warnings.filterwarnings("ignore")


# ==================================================
# 다른 폴더의 모듈을 가져오기 위한 설정
# ==================================================

# 현재 test_migration.py 파일이 있는 폴더입니다.
CURRENT_DIR = os.path.dirname(__file__)

# rag.py와 graph.py가 있는 rag_app 폴더 경로입니다.
RAG_APP_DIR = os.path.join(CURRENT_DIR, "..", "rag_app")

# Python이 rag_app 폴더의 파일을 찾을 수 있도록 경로를 추가합니다.
sys.path.insert(0, RAG_APP_DIR)


# ==================================================
# 비교할 두 가지 RAG 모듈 가져오기
# ==================================================

# 15차시에서 만든 일반 함수 방식입니다.
import rag

# 22차시에서 만든 LangGraph 방식입니다.
import graph22


# ==================================================
# 테스트 질문 목록
# ==================================================
TESTS = [
    # 정상적인 문서 질문입니다.
    "환불은 며칠 이내에 신청해야 하나요?",

    # 문서에 있을 가능성이 있는 질문입니다.
    "교환 기간은 얼마인가요?",

    # 문서에 있을 가능성이 있는 질문입니다.
    "고객센터 운영 시간은?",

    # 문서에 없는 질문입니다.
    "대표이사가 누구인가요?",

    # 표현을 바꾼 질문입니다.
    "반품하고 싶은데 언제까지?",

    # 여기에 추가 테스트 질문을 작성할 수 있습니다.
]


# ==================================================
# 테스트 결과를 비교하는 함수
# ==================================================
def run_test(question):
    """
    하나의 질문을 두 방식으로 실행하고 결과를 비교합니다.
    """

    # 15차시 일반 함수 방식으로 답변을 생성합니다.
    function_result = rag.ask(question)

    # 22차시 LangGraph 방식으로 답변을 생성합니다.
    graph_result = graph22.ask(question)

    # 각 결과에서 답변만 꺼냅니다.
    function_answer = function_result["answer"]
    graph_answer = graph_result["answer"]

    # 답변 앞뒤의 공백을 제거합니다.
    function_answer = function_answer.strip()
    graph_answer = graph_answer.strip()

    # 두 답변이 같은지 비교합니다.
    if function_answer == graph_answer:
        return True, function_answer, graph_answer

    # 두 답변이 다르면 불일치 결과를 반환합니다.
    return False, function_answer, graph_answer


# ==================================================
# 전체 테스트 실행
# ==================================================
def main():
    """
    모든 테스트 질문을 실행하고 최종 결과를 출력합니다.
    """

    # 답변이 일치한 질문의 개수입니다.
    same_count = 0

    # 답변이 다른 질문을 저장할 리스트입니다.
    different_questions = []

    # 테스트 질문을 하나씩 꺼냅니다.
    for question in TESTS:
        # 현재 질문을 두 방식으로 비교합니다.
        is_same, function_answer, graph_answer = run_test(question)

        # 답변이 같은 경우입니다.
        if is_same:
            same_count += 1
            print("✓", question[:28])

        # 답변이 다른 경우입니다.
        else:
            different_questions.append(question)

            print("✗", question[:28])
            print("    [함수]  ", function_answer[:60], "...")
            print("    [그래프]", graph_answer[:60], "...")

    # 전체 테스트 결과를 출력합니다.
    print()
    print("일치:", same_count, "/", len(TESTS))

    # 불일치한 질문이 있으면 출력합니다.
    if different_questions:
        print("불일치 질문:")

        for question in different_questions:
            print("-", question)


# ==================================================
# 이 파일을 직접 실행했을 때 테스트 시작
# ==================================================
if __name__ == "__main__":
    main()