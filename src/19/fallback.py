# ============================================================
# Fallback Node
# 검색 결과가 없을 때 사용자에게 안내 메시지를 반환합니다.
# 필요하면 지금까지 시도한 검색어도 함께 보여줍니다.
# ============================================================

import config


# ============================================================
# Fallback Node
# ============================================================
def fallback_node(state: dict) -> dict:
    """검색에 실패했을 때 안내 메시지를 만듭니다."""

    # 지금까지 시도한 검색어를 가져옵니다.
    tried_queries = state.get("tried_queries", [])

    # 기본 안내 메시지를 가져옵니다.
    message = config.MSG_NO_DOC

    # 시도한 검색어가 있으면 안내 메시지에 추가합니다.
    if tried_queries:
        tried_text = ", ".join(tried_queries)
        message += f"\n(시도한 검색어: {tried_text})"

    # 사용자가 다시 질문할 수 있도록 안내합니다.
    message += "\n질문을 조금 더 구체적으로 바꿔 다시 물어봐 주세요."

    # 변경할 State 값만 반환합니다.
    return {
        "answer": message,
        "grade": "giveup",
        "log": ["[fallback] 안내 메시지 반환"],
    }


# ============================================================
# Fallback Node 단독 테스트
# ============================================================
if __name__ == "__main__":

    # 테스트용 State를 만듭니다.
    state = {
        "question": "대표이사 이름은?"
    }

    # 그래프 없이 Fallback Node만 직접 실행합니다.
    result = fallback_node(state)

    # 만들어진 안내 메시지를 출력합니다.
    print(result["answer"])