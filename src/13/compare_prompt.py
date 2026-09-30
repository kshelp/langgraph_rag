import os
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

if __name__ == "__main__":
    compare("환불 신청 방법과 수수료를 알려주세요")

