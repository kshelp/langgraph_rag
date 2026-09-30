import os
import sys
import warnings

CURRENT_DIR = os.path.dirname(__file__)
INDEXER_DIR = os.path.join(CURRENT_DIR, "..", "10")

sys.path.insert(0, INDEXER_DIR)

from indexer import get_store

store = get_store()

QUESTION = "환불과 교환은 어떻게 다른가요?"

MIN_SCORE = 0.0

warnings.filterwarnings("ignore")

def compare_k(question, ks=(1, 3, 5)):

    print(f"질문: {question}")
    print("=" * 58)

    for k in ks:

        pairs = store.similarity_search_with_relevance_scores(
            question,
            k=k
        )

        scores = []

        low_count = 0

        for doc, score in pairs:

            scores.append(score)

            if score < MIN_SCORE:
                low_count = low_count + 1

        rounded_scores = []

        for score in scores:
            rounded_score = round(float(score), 3)
            rounded_scores.append(rounded_score)

        print(f"k={k:<3} 점수={rounded_scores}")

        print(f"     기준 미달({MIN_SCORE}) 조각: {low_count}개")
        print()

def show_result(name, retriever):
    docs = retriever.invoke(QUESTION)
    print(f"{len(docs)}개")

    for doc in docs:
        preview = doc.page_content[:55]
        preview.replace("\n", " ")
        page_no = doc.metadata["page_no"]
        print(f"{page_no} {preview}")
    print()

# similarity_retriever = store.as_retriever(
#     search_type = "similarity",
#     search_kwargs = {
#         "k":3
#     }
# )

# show_result("similarity k=3", similarity_retriever)

# mmr_retriever = store.as_retriever(
#     search_type = "mmr",
#     search_kwargs = {
#         "k":3
#     }
# )

# show_result("mmr k=3", mmr_retriever)

# compare_k("환불은 며칠 이내인가요?")
# compare_k("환불과 교환은 어떻게 다른가요?")

manual_retriever = store.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k":3,
        "filter": {
            "filename": "manual.pdf"
        }
    }
)

show_result("메뉴얼 필터", manual_retriever)
