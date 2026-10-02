# 이 파일은 질문과 관련된 문서를 검색하는 Retriever Node입니다.
# 재시도할수록 검색 문서 수를 늘리고, 점수 기준을 조금씩 완화합니다.

import os
import sys
import warnings

from dotenv import load_dotenv


# ==================================================
# 기본 경로와 환경 설정
# ==================================================

# 현재 파일이 있는 폴더의 절대 경로입니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 현재 파일의 상위 폴더인 src의 경로입니다.
SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

# src 폴더에서 config와 rag_app을 찾을 수 있도록 등록합니다.
sys.path.insert(0, SRC_DIR)

# 불필요한 경고 메시지를 숨깁니다.
warnings.filterwarnings("ignore")

# 환경 변수 파일을 불러옵니다.
load_dotenv()

# 프로젝트 설정을 불러옵니다.
import config

# 벡터 저장소를 가져오는 함수를 불러옵니다.
from rag_app.indexer import get_store


# ==================================================
# 벡터 저장소 준비
# ==================================================

# 모듈이 처음 실행될 때 벡터 저장소를 한 번만 준비합니다.
_store = get_store()


# ==================================================
# 검색 방식에 따라 문서를 검색합니다.
# ==================================================
def search_documents(query: str, k: int, search_type: str):
    """
    질문에 맞는 문서를 검색합니다.

    search_type이 "mmr"이면 MMR 검색을 사용하고,
    그 외의 경우에는 Similarity 검색을 사용합니다.
    """

    # MMR 검색을 사용하는 경우입니다.
    if search_type == "mmr":
        documents = _store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k * 4,
            lambda_mult=0.5,
        )

        # MMR은 점수를 제공하지 않으므로 None을 넣습니다.
        return [
            (document, None)
            for document in documents
        ]

    # 기본 검색 방식인 Similarity 검색입니다.
    return _store.similarity_search_with_relevance_scores(
        query,
        k=k,
    )


# ==================================================
# 재시도 가능한 Retriever Node
# ==================================================
def retriever_node(state: dict) -> dict:
    """
    State에서 검색 조건을 읽고 문서를 검색합니다.

    재시도 횟수에 따라 검색 범위와 점수 기준을 조정합니다.
    """

    # 현재까지 재시도한 횟수를 가져옵니다.
    retries = state.get("retries", 0)

    # query가 있으면 query를 사용합니다.
    query = state.get("query")

    # query가 없으면 question을 검색어로 사용합니다.
    if not query:
        query = state.get("question", "")

    # State에 top_k가 있으면 그 값을 사용합니다.
    top_k = state.get("top_k")

    # top_k가 없으면 재시도 횟수에 따라 검색 개수를 늘립니다.
    if not top_k:
        top_k = config.TOP_K + retries * 3

    # State에 search_type이 있으면 그 값을 사용합니다.
    search_type = state.get("search_type")

    # search_type이 없으면 config의 기본값을 사용합니다.
    if not search_type:
        search_type = config.SEARCH_TYPE

    # State에 min_score가 있으면 그 값을 사용합니다.
    min_score = state.get("min_score")

    # min_score가 없으면 재시도 횟수에 따라 기준을 완화합니다.
    if min_score is None:
        min_score = config.MIN_SCORE - retries * 0.05

        # 점수 기준이 0.0보다 낮아지지 않도록 제한합니다.
        if min_score < 0.0:
            min_score = 0.0

    # 질문이 비어 있으면 검색하지 않습니다.
    if not query.strip():
        return {
            "documents": [],
            "scores": [],
            "retrieval_ok": False,
            "fail_reason": "empty_query",
            "log": ["검색: 질의가 비어 있음"],
        }

    # 검색 중 오류가 발생해도 그래프가 중단되지 않도록 처리합니다.
    try:
        search_results = search_documents(
            query,
            top_k,
            search_type,
        )

    except Exception as error:
        return {
            "documents": [],
            "scores": [],
            "retrieval_ok": False,
            "fail_reason": "search_error",
            "log": [
                f"검색 오류: {type(error).__name__}"
            ],
        }

    # 검색 결과가 없으면 정상적인 실패 상태를 반환합니다.
    if not search_results:
        return {
            "documents": [],
            "scores": [],
            "retrieval_ok": False,
            "fail_reason": "no_result",
            "log": ["검색: 결과 0건"],
        }

    # 검색 결과의 점수를 저장합니다.
    scores = []

    for document, score in search_results:
        # MMR은 점수가 없으므로 None을 저장합니다.
        if score is None:
            scores.append(None)

        # Similarity 검색 점수는 소수 셋째 자리까지 저장합니다.
        else:
            scores.append(round(float(score), 3))

    # 최소 점수를 통과한 문서를 저장합니다.
    kept_documents = []

    # Similarity 검색은 점수 기준으로 문서를 필터링합니다.
    if search_results[0][1] is not None:

        for document, score in search_results:
            # 최소 점수 이상인 문서만 보관합니다.
            if float(score) >= float(min_score):
                kept_documents.append(document)

        # 통과한 문서가 없으면 실패 이유를 기록합니다.
        if kept_documents:
            fail_reason = ""
        else:
            fail_reason = "low_score"

    # MMR 검색은 점수가 없으므로 결과를 그대로 사용합니다.
    else:
        for document, score in search_results:
            kept_documents.append(document)

        # MMR은 점수 기준으로 실패 여부를 판단하지 않습니다.
        fail_reason = ""

    # 최종 검색 결과를 반환합니다.
    return {
        "documents": kept_documents,
        "scores": scores,
        "retrieval_ok": len(kept_documents) > 0,
        "fail_reason": fail_reason,
        "log": [
            f"검색 방식={search_type}, "
            f"재시도={retries}, "
            f"k={top_k}, "
            f"기준={min_score}, "
            f"{len(kept_documents)}/{len(search_results)}건 통과"
        ],
    }


# ==================================================
# 재시도 횟수에 따른 동작 테스트
# ==================================================
if __name__ == "__main__":

    # 반복해서 검색할 질문입니다.
    query = "환불은 며칠 이내인가요?"

    # 재시도 횟수를 0회, 1회, 2회로 바꾸어 테스트합니다.
    for retries in range(3):

        # Retriever Node에 전달할 State입니다.
        state = {
            "query": query,
            "retries": retries,
        }

        # Retriever Node를 실행합니다.
        result = retriever_node(state)

        # 검색 결과를 출력합니다.
        print(
            f"retries={retries} | "
            f"ok={result['retrieval_ok']} | "
            f"사유={result['fail_reason'] or '-'}"
        )

        # 실제 적용된 검색 과정을 출력합니다.
        print(
            f"   {result['log'][0]}"
        )