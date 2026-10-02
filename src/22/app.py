# 이 파일은 사용자의 질문을 받아 LangGraph 기반 RAG를 실행합니다.
# 실행 결과로 답변과 답변에 사용된 문서 출처를 화면에 출력합니다.

import os
import sys


# ==================================================
# graph.py 파일을 찾기 위한 경로 설정
# ==================================================

# 현재 app.py 파일이 있는 폴더입니다.
CURRENT_DIR = os.path.dirname(__file__)

# graph.py가 있는 rag_app 폴더의 경로입니다.
RAG_APP_DIR = os.path.join(
    CURRENT_DIR,
    "..",
    "rag_app",
)

# Python이 rag_app 폴더의 모듈을 찾도록 경로를 추가합니다.
sys.path.insert(0, RAG_APP_DIR)


# LangGraph 기반 RAG의 ask() 함수를 가져옵니다.
from graph import ask


# ==================================================
# 질문을 받고 답변을 출력하는 함수
# ==================================================
def main():
    """
    사용자에게 질문을 받고 RAG 답변과 출처를 출력합니다.
    """

    # 사용자에게 질문을 입력받습니다.
    question = input("질문을 입력하세요: ")

    # 입력받은 질문을 RAG 시스템에 전달합니다.
    result = ask(question)

    # 최종 답변을 출력합니다.
    print()
    print("답변:")
    print(result["answer"])

    # 답변에 사용된 출처를 출력합니다.
    print()
    print("출처:")

    # 출처를 하나씩 꺼내 출력합니다.
    for source in result["sources"]:
        print(source)


# ==================================================
# 이 파일을 직접 실행했을 때 시작합니다.
# ==================================================
if __name__ == "__main__":
    main()