import os
import re
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

CURRENT_DIR = os.path.dirname(__file__)
RETRIEVER_DIR = os.path.join(CURRENT_DIR, "..", "11")

sys.path.append(RETRIEVER_DIR)

from retriever import search, build_context
from prompts import PROMPTS

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def check_citation(answer, document_count):

    # 답변에서 [숫자] 형태의 인용 번호를 모두 찾습니다.
    found_numbers = re.findall(r"\[(\d+)\]", answer)

    # 인용 번호가 하나도 없으면 실패입니다.
    if not found_numbers:
        return False, "인용 번호가 전혀 없음"

    # 찾은 문자열 번호를 정수로 바꿉니다.
    citation_numbers = []

    for number in found_numbers:
        citation_numbers.append(int(number))

    # 존재하지 않는 인용 번호가 있는지 검사합니다.
    invalid_numbers = []

    for number in citation_numbers:

        # 인용 번호는 1부터 검색 문서 개수까지만 가능합니다.
        if number < 1 or number > document_count:
            invalid_numbers.append(number)

    # 잘못된 인용 번호가 있으면 실패입니다.
    if invalid_numbers:
        return False, f"존재하지 않는 근거 번호: {invalid_numbers}"

    # 모든 인용 번호가 정상입니다.
    return True, f"인용 {len(citation_numbers)}건 정상"

def compare(question):

    documents = search(question)
    context = build_context(documents)

    print("=" * 60)
    print("Q:", question)
    print("근거:", len(documents), "개")
    print("-" * 60)

    for name, prompt in PROMPTS.items():
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        print()
        print(f"[{name}]")
        print(answer)

        citation_ok, citation_message = check_citation(
            answer,
            len(documents)
        )

        if citation_ok:
            mark = "✓"
        else:
            mark = "✗"

        print("🔍 인용 검증:", mark, citation_message)


if __name__ == "__main__":
     
    test_questions = [
        "환불 신청 방법과 수수료를 알려주세요",
        "환불은 언제까지 가능한가요?",
        "배송 비용은 얼마인가요?"
    ]

    for question in test_questions:
        compare(question)
        print()
