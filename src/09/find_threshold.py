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

BASE = "환불은 상품 수령 후 7일 이내에 신청할 수 있습니다"

RELATED = [
    "반품하고 싶어요",
    "돈 돌려받을 수 있나요?",
    "구매 취소 기간이 어떻게 되나요?",
    "물건 안 마음에 들면 어떡하죠?",
    "환불 신청 기한 알려주세요",
]

UNRELATED = [
    "회사 주차장은 어디인가요?",
    "채용 공고 보고 싶어요",
    "오늘 날씨 어때요?",
    "대표이사가 누구인가요?",
    "직원 복지 제도 알려주세요",
]

related_scores = []

for question in RELATED:
    score = sim(BASE, question)
    related_scores.append(score)

unrelated_scores = []

for question in UNRELATED:
    score = sim(BASE, question)
    unrelated_scores.append(score)

for i in range(len(RELATED)):
    print(f"   {related_scores[i]:.3f}  {RELATED[i]}")

for i in range(len(UNRELATED)):
    print(f"   {unrelated_scores[i]:.3f}  {UNRELATED[i]}")

# 임계값 후보 계산
min_related = min(related_scores)
max_unrelated = max(unrelated_scores)

threshold = (min_related + max_unrelated) / 2

print(f"\n 임계값 후보 :  {threshold : .2f}")


