# 이 코드는 LangGraph에서 사용할 상태와 더미 노드를 정의합니다.
# 실제 검색이나 LLM 호출 없이 검색, 생성, 검증, 재시도 과정을 흉내 냅니다.

from typing import TypedDict, List, Annotated
import operator


# ============================================================
# 그래프 상태 정의
# ============================================================

class MiniState(TypedDict):
    # 사용자가 입력한 질문
    question: str

    # 검색된 문서 목록
    documents: List[str]

    # 생성된 답변
    answer: str

    # 검증 결과
    # pass: 통과, retry: 재시도, giveup: 포기
    grade: str

    # 재시도 횟수
    retries: int

    # 노드 실행 기록
    # 각 노드에서 만든 로그를 계속 누적합니다.
    log: Annotated[List[str], operator.add]


# ============================================================
# 초기 상태 생성 함수
# ============================================================

def init(question: str) -> MiniState:
    """
    그래프 실행에 사용할 초기 상태를 만듭니다.

    처음에는 검색 결과와 답변이 없으므로
    관련 항목을 빈 값으로 설정합니다.
    """

    # 그래프 실행에 사용할 초기 상태를 반환합니다.
    return {
        "question": question,
        "documents": [],
        "answer": "",
        "grade": "",
        "retries": 0,
        "log": [],
    }


# ============================================================
# Retriever Node
# ============================================================

def retrieve(state: MiniState) -> dict:
    """
    질문을 바탕으로 검색 결과를 만듭니다.

    질문에 '없는'이라는 단어가 포함되면
    검색 실패 상황을 흉내 냅니다.
    """

    # 현재 질문을 가져옵니다.
    question = state["question"]

    # 검색 결과를 저장할 리스트를 만듭니다.
    documents = []

    # 질문에 '없는'이 포함되어 있는지 확인합니다.
    if "없는" in question:
        # 검색 실패 상황을 만듭니다.
        documents = []
    else:
        # 검색 성공 상황을 만듭니다.
        documents = ["조각A", "조각B", "조각C"]

    # 검색된 문서 개수를 출력합니다.
    print(f"   → retrieve: {len(documents)}건")

    # 변경된 값만 반환합니다.
    return {
        "documents": documents,
        "log": [f"검색 {len(documents)}건"],
    }


# ============================================================
# Generator Node
# ============================================================

def generate(state: MiniState) -> dict:
    """
    검색된 문서를 바탕으로 가짜 답변을 생성합니다.

    첫 번째 답변은 부실하게 만들고,
    재시도 후에는 좋은 답변을 만듭니다.
    """

    # 검색된 문서의 개수를 확인합니다.
    document_count = len(state["documents"])

    # 현재 재시도 횟수를 확인합니다.
    retry_count = state["retries"]

    # 답변 품질을 저장할 변수입니다.
    quality = ""

    # 재시도 횟수에 따라 답변 품질을 결정합니다.
    if retry_count >= 1:
        # 재시도 후에는 좋은 답변을 만듭니다.
        quality = "좋은"
    else:
        # 첫 번째 답변은 부실한 답변으로 처리합니다.
        quality = "부실한"

    # 생성된 답변의 품질을 출력합니다.
    print(f"   → generate: {quality} 답변")

    # 가짜 답변을 생성합니다.
    answer = f"{document_count}건 근거로 만든 {quality} 답변"

    # 변경된 값만 반환합니다.
    return {
        "answer": answer,
        "log": [f"생성({quality})"],
    }


# ============================================================
# Verifier Node
# ============================================================

def verify(state: MiniState) -> dict:
    """
    생성된 답변을 검증합니다.

    답변에 '좋은'이라는 단어가 있으면 통과시키고,
    없으면 재시도 대상으로 처리합니다.
    """

    # 현재 답변을 가져옵니다.
    answer = state["answer"]

    # 검증 결과를 저장할 변수를 만듭니다.
    grade = ""
    message = ""

    # 답변에 '좋은'이라는 단어가 포함되어 있는지 확인합니다.
    if "좋은" in answer:
        # 좋은 답변이면 통과시킵니다.
        grade = "pass"
        message = "통과"
    else:
        # 부실한 답변이면 재시도 대상으로 처리합니다.
        grade = "retry"
        message = "실패"

    # 검증 결과를 출력합니다.
    if grade == "pass":
        print("   → verify: 통과")
    else:
        print("   → verify: 재시도")

    # 검증 결과와 로그를 반환합니다.
    return {
        "grade": grade,
        "log": [f"검증 {message}"],
    }


# ============================================================
# Retry Counter Node
# ============================================================

def bump(state: MiniState) -> dict:
    """
    재시도 횟수를 1 증가시킵니다.
    """

    # 현재 재시도 횟수에 1을 더합니다.
    retry_count = state["retries"] + 1

    # 증가한 재시도 횟수를 출력합니다.
    print(f"   → bump: {retry_count}회차")

    # 변경된 값만 반환합니다.
    return {
        "retries": retry_count,
        "log": [f"재시도 {retry_count}"],
    }


# ============================================================
# Fallback Node
# ============================================================

def fallback(state: MiniState) -> dict:
    """
    검색에 실패했거나 더 이상 처리할 수 없을 때
    사용자에게 안내 답변을 반환합니다.
    """

    # Fallback 노드가 실행되었음을 출력합니다.
    print("   → fallback")

    # 안내 답변과 포기 상태를 반환합니다.
    return {
        "answer": "자료를 찾지 못했습니다.",
        "grade": "giveup",
        "log": ["fallback"],
    }