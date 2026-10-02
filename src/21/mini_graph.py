# 이 코드는 LangGraph에서 사용할 상태(State)의 구조를 정의합니다.
# 질문, 검색 문서, 답변, 검증 결과, 재시도 횟수, 실행 로그를 관리합니다.

from typing import TypedDict, List, Annotated
import operator


# ------------------------------------------------------------
# 그래프가 공유할 상태 구조
# ------------------------------------------------------------

class MiniState(TypedDict):
    # 사용자가 입력한 질문
    question: str

    # 검색된 문서 목록
    documents: List[str]

    # 생성된 답변
    answer: str

    # 답변 검증 결과
    # 예: "pass", "retry", "giveup"
    grade: str

    # 현재까지 재시도한 횟수
    retries: int

    # 그래프 실행 과정을 기록하는 로그
    # 여러 노드의 로그를 하나의 리스트로 합칩니다.
    log: Annotated[List[str], operator.add]


# ------------------------------------------------------------
# 그래프 실행에 사용할 초기 상태 만들기
# ------------------------------------------------------------

def init(question: str) -> MiniState:
    """
    사용자의 질문을 바탕으로 초기 상태를 만듭니다.

    그래프를 처음 실행할 때는 아직 검색 결과와 답변이 없습니다.
    따라서 documents, answer, grade, log는 빈 값으로 시작합니다.
    retries도 처음에는 0으로 설정합니다.
    """

    # 그래프 실행에 필요한 초기 상태를 반환합니다.
    return {
        "question": question,
        "documents": [],
        "answer": "",
        "grade": "",
        "retries": 0,
        "log": [],
    }