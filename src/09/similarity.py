from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

emb = OpenAIEmbeddings(model="text-embedding-3-small")

def sim(sentence1, sentence2):
    vector1 = emb.embed_query(sentence1)
    vector2 = emb.embed_query(sentence2)

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    similarity = dot_product / (length1 * length2)
    return float(similarity)

PAIRS = [
    ("환불 규정", "반품 절차"),
    ("비밀번호 변경", "패스워드 재설정"),
    ("제품이 고장났어요", "기기 불량 신고"),
    ("환불 규정", "배송 안내"),
    ("환불 규정", "오늘 점심 메뉴"),
]

for sentence1, sentence2 in PAIRS:
    score = sim(sentence1, sentence2)
    print(f"{score : .3f} {sentence1} <-> {sentence2}")

