from langchain_core.prompts import ChatPromptTemplate

# ── V1: 최소 버전 (비교 대조군) ──
RAG_PROMPT_V1 = ChatPromptTemplate.from_template(     
    "아래 자료를 참고해서 질문에 답해주세요.\n\n"     
    "[자료]\n{context}\n\n"
    "[질문] {question}"
)
# ── V2: 규칙 강화 버전 (권장) ──
RAG_PROMPT_V2 = ChatPromptTemplate.from_template(     
    "당신은 사내 문서를 안내하는 담당자입니다.\n\n"
    "[규칙]\n"
    "1. 아래 [자료]에 있는 내용만 근거로 답하십시오.\n"     
    "2. 자료에 없는 내용은 '자료에서 확인할 수 없습니다'라고 "     
    "명시하십시오.\n"
    "3. 추측하거나 일반 상식으로 보충하지 마십시오.\n"     
    "4. 각 문장 끝에 근거 번호를 [1] 형식으로 표기하십시오.\n"     
    "5. 3~5문장으로 간결하게 답하십시오.\n\n"     
    "[자료]\n{context}\n\n"
    "[질문]\n{question}"
)
# ── V3: 부분 답변 허용 버전 (과잉 거부 방지) ──
RAG_PROMPT_V3 = ChatPromptTemplate.from_template(     
    "당신은 사내 문서를 안내하는 담당자입니다.\n\n"
    "[규칙]\n"
    "1. 아래 [자료]에 있는 내용만 근거로 답하십시오.\n"     
    "2. 질문의 일부만 자료에 있다면, 있는 부분은 답하고 "     
    "없는 부분만 '자료에서 확인할 수 없습니다'라고 하십시오.\n"     
    "3. 추측하거나 일반 상식으로 보충하지 마십시오.\n"     
    "4. 각 문장 끝에 근거 번호를 [1] 형식으로 표기하십시오.\n"     
    "5. 3~5문장으로 간결하게 답하십시오.\n\n"     
    "[자료]\n{context}\n\n"
    "[질문]\n{question}"
)

PROMPTS = {"v1": RAG_PROMPT_V1,            
           "v2": RAG_PROMPT_V2,            
           "v3": RAG_PROMPT_V3}



# 25차시에 추가

RETRY_PROMPT = ChatPromptTemplate.from_template(
    "당신은 사내 문서를 안내하는 담당자입니다.\n\n"
    "[이전 시도 결과]\n"
    "직전 답변이 아래 이유로 반려되었습니다.\n"
    "{reason}\n"
    "이번에는 반드시 이 문제를 해결해서 작성하십시오.\n\n"
    "[규칙]\n"
    "1. 아래 [자료]에 있는 내용만 근거로 답하십시오.\n"
    "2. 질문의 일부만 자료에 있다면 있는 부분만 답하십시오.\n"
    "3. 각 문장 끝에 근거 번호를 [1] 형식으로 반드시 표기하십시오.\n"
    "4. 반드시 한 문장으로 간결하게 답하십시오.\n\n"
    "[자료]\n"
    "{context}\n\n"
    "[질문]\n"
    "{question}"
)

# PROMPTS["retry"] = RETRY_PROMPT


# 26차시에 추가

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "당신은 RAG 답변을 검증하는 엄격한 심판입니다.\n"
    "답변을 생성하지 말고, 오직 평가만 하십시오.\n\n"
    "[근거 자료]\n{context}\n\n"
    "[사용자 질문]\n{question}\n\n"
    "[검증할 답변]\n{answer}\n\n"
    "[평가 기준]\n"
    "grounded : 답변의 모든 주장이 근거 자료 안에 있으면 true.\n"     
    "           근거에 없는 숫자·조건·설명이 하나라도 있으면 false.\n"     
    "relevant : 답변이 질문에 실제로 대답하면 true.\n"
    "           질문의 핵심을 비껴갔으면 false.\n"
    "reason   : 판정 이유를 한 문장으로. false인 항목이 있으면\n"     
    "           어느 부분이 문제인지 구체적으로 쓸 것.\n\n"
    "[출력 형식 - 다른 말은 절대 쓰지 마십시오]\n"
    '{{"grounded": true, "relevant": true, "reason": "..."}}' )

# 29차시 추가
CLASSIFY_PROMPT = ChatPromptTemplate.from_template(     
    "사용자 질문의 유형을 하나만 골라 그 단어만 출력하십시오.\n"
    "다른 설명은 절대 하지 마십시오.\n\n"
    "[유형]\n"
    "greeting : 인사, 감사, 잡담 (정보 요청이 아님)\n"
    "calc     : 순수한 수식 계산\n"
    "scope    : 우리 문서와 무관한 주제 "
    "(날씨, 뉴스, 일반 상식 등)\n"
    "document : 사내 문서에서 찾아야 할 질문\n\n"
    "[판단 원칙]\n"
    "애매하면 document를 선택하십시오.\n\n"     
    "[질문]\n{question}\n\n"
    "[유형]" 
    )

# 30차시 추가
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "당신은 문서 검색을 돕는 질의 재작성 전문가입니다.\n"
    "아래 질문으로 문서를 검색했지만 관련 자료를 찾지 못했습니다.\n"
    "검색이 더 잘 되도록 질의를 다시 작성하십시오.\n\n"
    "[원본 질문]\n{question}\n\n"
    "[이미 시도한 질의 - 반복하지 마십시오]\n{tried}\n\n"
    "[실패 이유]\n{reason}\n\n"
    "[재작성 규칙]\n"
    "1. 일상적 표현을 문서에서 쓸 법한 공식 용어로 바꾸십시오.\n"
    "   예) 반품 -> 환불 / 택배비 -> 배송비 / 비번 -> 비밀번호\n"
    "2. 구어체를 문어체로 바꾸십시오.\n"
    "3. 군더더기를 빼고 핵심 명사 위주로 3~8단어로 압축하십시오.\n"     
    "4. 원본의 의미를 바꾸지 마십시오.\n"
    "5. 이미 시도한 질의와 다르게 작성하십시오.\n\n"     
    "[출력 - 재작성된 질의만 한 줄로. 다른 말 금지]" 
    )