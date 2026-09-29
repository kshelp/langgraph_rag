from langchain_core.prompts import PromptTemplate

template = """
다음 문서를 읽고 질문에 답하세요.
문서: {doc}
질문: {q}
답변: 
"""

prompt = PromptTemplate(
    input_variables=["doc", "q"],
    template=template
)

result = prompt.format(
    doc = "제품 색상은 검은색입니다.",
    q = "제품 색상은?"
)

print(result)

