# 이 파일은 22차시 그래프와 27차시 그래프의 답변 품질을 비교합니다.
# 정답 키워드 포함 여부와 문서에 없는 질문에 대한 환각 여부를 확인합니다.


import os
import sys


# ============================================================
# 필요한 폴더 등록
# ============================================================

# 현재 파일이 있는 폴더의 경로입니다.
BASE_DIR = os.path.dirname(__file__)

# 각 차시 폴더를 Python 모듈 검색 경로에 추가합니다.
sys.path.insert(0, os.path.join(BASE_DIR, "..", "18"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "22"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "24"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "25"))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "26"))


# 비교할 두 그래프를 가져옵니다.
import graph22
import graph27


# ============================================================
# 테스트 질문
# ============================================================

TESTS = [
       {
        "q": "환불 방법을 알려주세요.",
        "keys": ["환불"],
    },
    {
        "q": "반품은 언제까지 가능한가요?",
        "keys": ["7일", "반품"],
    },
    {
        "q": "상품을 교환할 수 있나요?",
        "keys": ["교환"],
    },
    {
        "q": "환불 방법과 대표이사를 알려주세요",
        "keys": ["환불"],
    },
    

    # 문서에 없는 질문입니다.
    {
        "q": "대표이사는 누구인가요?",
        "no_answer": True,
    },
    {
        "q": "회사 창립일은 언제인가요?",
        "no_answer": True,
    },

]


# ============================================================
# 답변할 근거가 없다는 내용인지 확인
# ============================================================

def is_no_answer(answer):
    """답변이 '알 수 없다'는 안내인지 확인합니다."""

    # 근거가 없을 때 사용할 수 있는 표현들입니다.
    no_answer_words = [
        "확인할 수 없",
        "찾지 못",
        "관련 자료가 없",
        "알 수 없",
        "제공된 자료",
    ]

    # 안내 문구가 하나라도 답변에 있으면 True를 반환합니다.
    for word in no_answer_words:
        if word in answer:
            return True

    # 안내 문구가 없으면 False를 반환합니다.
    return False


# ============================================================
# 정답 키워드 포함 여부 확인
# ============================================================

def contains_answer_keyword(answer, test):
    """답변에 정답 키워드가 포함되어 있는지 확인합니다."""

    # 답변할 근거가 없다는 안내가 나오면 정답으로 보지 않습니다.
    if is_no_answer(answer):
        return False

    # 테스트 질문에 등록된 정답 키워드를 하나씩 확인합니다.
    for key in test.get("keys", []):

        # 키워드가 답변에 있으면 정답으로 판단합니다.
        if key in answer:
            return True

    # 어떤 키워드도 없으면 오답으로 판단합니다.
    return False


# ============================================================
# 하나의 그래프 평가
# ============================================================

def evaluate(graph_module, graph_name):
    """하나의 그래프에 모든 테스트 질문을 실행합니다."""

    # 정답으로 판단된 질문 수입니다.
    hit_count = 0

    # 근거 없이 답변한 것으로 의심되는 질문 수입니다.
    hallucination_count = 0

    # 그래프 이름을 먼저 출력합니다.
    print()
    print("=" * 50)
    print(f"{graph_name} 평가 시작")
    print("=" * 50)

    # 모든 테스트 질문을 하나씩 실행합니다.
    for test in TESTS:

        # 현재 질문을 그래프에 전달합니다.
        result = graph_module.ask(test["q"])

        # 결과에서 답변을 가져옵니다.
        answer = result.get("answer", "")

        # 현재 질문이 문서에 없는 질문인지 확인합니다.
        no_answer_question = test.get("no_answer", False)

        # 현재 질문의 정답 여부를 저장합니다.
        is_correct = False

        # --------------------------------------------
        # 문서에 없는 질문 처리
        # --------------------------------------------

        if no_answer_question:

            # 모른다고 안내하면 올바른 대응입니다.
            if is_no_answer(answer):
                is_correct = True

            # 모른다고 하지 않고 구체적인 답변을 하면 환각 의심입니다.
            else:
                hallucination_count += 1

        # --------------------------------------------
        # 문서에 있는 질문 처리
        # --------------------------------------------

        else:

            # 정답 키워드가 포함되었는지 확인합니다.
            is_correct = contains_answer_keyword(answer, test)

        # 정답이면 전체 정답 수를 증가시킵니다.
        if is_correct:
            hit_count += 1

        # 질문별 실행 결과를 출력합니다.
        print()
        print("질문:", test["q"])
        print("답변:", answer[:100])
        print("판정:", result.get("grade", ""))
        print("재시도:", result.get("retries", 0))
        print("정답 여부:", is_correct)

    # 전체 질문 수를 계산합니다.
    total_count = len(TESTS)

    # 정답률을 계산합니다.
    accuracy = hit_count / total_count * 100

    # 최종 평가 결과를 출력합니다.
    print()
    print("=" * 50)
    print(f"[{graph_name}]")
    print(f"정답: {hit_count}/{total_count}")
    print(f"정답률: {accuracy:.0f}%")
    print(f"환각 의심: {hallucination_count}건")
    print("=" * 50)


# ============================================================
# 프로그램 실행
# ============================================================

if __name__ == "__main__":

    # 22차시 검증 전 그래프를 평가합니다.
    evaluate(graph22, "22차시")

    # 27차시 검증 및 재생성 그래프를 평가합니다.
    evaluate(graph27, "27차시")