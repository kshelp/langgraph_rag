# ============================================================
# Retriever Node
# 사용자의 질문과 관련된 문서를 벡터 저장소에서 검색합니다.
# 기준 점수 이상인 문서만 골라 State에 저장할 결과를 반환합니다.
# ============================================================

import os
import sys

import config
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 10차시에 만든 indexer.py를 사용하기 위해 경로를 추가합니다.
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEXER_DIR = os.path.join(CURRENT_DIR, "..", "10")

sys.path.insert(0, INDEXER_DIR)

from indexer import get_store


# ------------------------------------------------------------
# 저장된 벡터 저장소를 불러옵니다.
# ------------------------------------------------------------
store = get_store()


# ============================================================
# Retriever Node
# ============================================================
def retrieve_node(state: dict) -> dict:
    """질문과 관련된 문서를 검색합니다."""

    # query가 있으면 사용하고, 없으면 원래 question을 사용합니다.
    query = state.get("query")

    if not query:
        query = state["question"]

    # top_k가 있으면 사용하고, 없으면 기본값을 사용합니다.
    top_k = state.get("top_k", config.TOP_K)

    # 질문과 유사한 문서를 검색합니다.
    results = store.similarity_search_with_relevance_scores(
        query,
        k=top_k
    )

    # 기준 점수 이상인 문서만 저장합니다.
    documents = []

    for document, score in results:
        if score >= config.MIN_SCORE:
            documents.append(document)

    # 검색된 모든 문서의 점수를 저장합니다.
    scores = []

    for document, score in results:
        scores.append(round(score, 3))

    # 기준을 통과한 문서가 있는지 확인합니다.
    retrieval_ok = len(documents) > 0

    # 실행 내용을 로그로 남깁니다.
    log_message = (
        f"[retrieve] '{query}' → "
        f"{len(documents)}/{len(results)}건 통과"
    )

    # 변경할 State 값만 반환합니다.
    return {
        "documents": documents,
        "scores": scores,
        "retrieval_ok": retrieval_ok,
        "log": [log_message],
    }


# ============================================================
# Retriever Node 단독 테스트
# ============================================================
if __name__ == "__main__":

    # 테스트용 State를 만듭니다.
    state = {
        "question": "환불은 며칠 이내인가요?"
    }

    # 그래프 없이 Retriever Node만 직접 실행합니다.
    result = retrieve_node(state)

    # 검색 결과를 확인합니다.
    print("통과 문서:", len(result["documents"]))
    print("점수      :", result["scores"])
    print("검색 성공 :", result["retrieval_ok"])
    print("로그      :", result["log"][0])