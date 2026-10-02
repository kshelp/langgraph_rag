# ============================================================
# Intent Nodes
# ============================================================
# 질문 유형에 따라 간단한 답변을 반환하는 노드 모음입니다.
# 인사말, 계산식, 문서 범위 밖 질문을 각각 처리합니다.
# ============================================================


import re


# ============================================================
# 1. 인사말 처리 노드
# ============================================================

def greeting_node(state):
    """
    사용자가 인사했을 때 안내 메시지를 반환합니다.
    """

    # 사내 문서 질문을 도와줄 수 있다는 안내문입니다.
    answer = (
        "안녕하세요! 사내 문서에 관한 질문을 도와드립니다.\n"
        "예: 환불 규정, 휴가 신청 방법, 고객센터 운영 시간 등"
    )

    # 답변과 처리 결과를 State에 저장합니다.
    return {
        "answer": answer,
        "grade": "pass",
        "reason": "인사말에 대한 안내 완료",
        "log": ["인사말 응답"],
    }


# ============================================================
# 2. 계산 처리 노드
# ============================================================

def calc_node(state):
    """
    사용자가 입력한 간단한 계산식을 계산합니다.
    """

    # State에서 사용자의 질문을 가져옵니다.
    question = state.get("question", "")

    # 질문 앞뒤의 공백을 제거합니다.
    expression = question.strip()

    # 마지막에 붙은 '=' 또는 '?'를 제거합니다.
    expression = expression.rstrip("=?").strip()

    # 숫자와 계산 기호만 허용합니다.
    allowed_pattern = r"[\d\s+\-*/().,]+"

    # 허용되지 않은 문자가 있으면 계산하지 않습니다.
    if re.fullmatch(allowed_pattern, expression) is None:
        return {
            "answer": "계산식을 처리하지 못했습니다.",
            "grade": "giveup",
            "reason": "허용되지 않은 문자가 있습니다.",
            "log": ["계산 실패"],
        }

    try:
        # 천 단위 구분 쉼표를 제거합니다.
        clean_expression = expression.replace(",", "")

        # 검사를 통과한 계산식을 계산합니다.
        result = eval(
            clean_expression,
            {
                "__builtins__": {}
            },
            {},
        )

        # 계산 결과를 반환합니다.
        return {
            "answer": f"{expression} = {result}",
            "grade": "pass",
            "reason": "계산 완료",
            "log": [
                f"계산 완료: {expression} = {result}"
            ],
        }

    except Exception as error:
        # 계산 중 오류가 발생하면 안내 메시지를 반환합니다.
        return {
            "answer": "계산식을 처리하지 못했습니다.",
            "grade": "giveup",
            "reason": f"계산 중 오류 발생: {type(error).__name__}",
            "log": ["계산 실패"],
        }


# ============================================================
# 3. 문서 범위 밖 질문 처리 노드
# ============================================================

def scope_node(state):
    """
    사내 문서와 관련 없는 질문에 안내 메시지를 반환합니다.
    """

    # 문서 질문만 처리한다는 안내문입니다.
    answer = (
        "죄송합니다.\n"
        "저는 사내 문서에 관한 질문에만 답변할 수 있습니다.\n"
        "문서와 관련된 내용을 물어봐 주세요."
    )

    # 범위 밖 질문에 대한 결과를 반환합니다.
    return {
        "answer": answer,
        "grade": "pass",
        "reason": "문서 범위 밖 질문에 대한 안내 완료",
        "log": ["문서 범위 밖 질문"],
    }


# ============================================================
# 단독 실행 테스트
# ============================================================

if __name__ == "__main__":

    # 각 노드를 테스트할 질문입니다.
    test_cases = [
        ("greeting", "안녕하세요"),
        ("calc", "1,000 + 2,000 * 3"),
        ("scope", "오늘 날씨 어때요?"),
    ]

    # 질문을 하나씩 테스트합니다.
    for intent, question in test_cases:

        # 노드에 전달할 State를 만듭니다.
        state = {
            "question": question
        }

        # 의도에 따라 실행할 노드를 선택합니다.
        if intent == "greeting":
            result = greeting_node(state)

        elif intent == "calc":
            result = calc_node(state)

        else:
            result = scope_node(state)

        # 실행 결과를 출력합니다.
        print(f"\n질문: {question}")
        print(f"답변: {result['answer']}")
        print(f"판정: {result['grade']}")
        print(f"로그: {result['log']}")