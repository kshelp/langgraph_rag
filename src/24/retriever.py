# 이 파일은 질문과 관련된 문서를 검색하는 Retriever Node입니다.
# Similarity 검색과 MMR 검색을 지원하며, 검색 점수와 실패 원인을 함께 반환합니다.

import os
import sys
import warnings

from dotenv import load_dotenv


# ==================================================
# 기본 환경 설정
# ==================================================

# 현재 파일의 위치를 기준으로 src 폴더를 찾습니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 현재 파일의 상위 폴더인 src를 계산합니다.
SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

# src 폴더에서 config와 rag_app을 찾을 수 있도록 등록합니다.
sys.path.insert(0, SRC_DIR)

# 불필요한 경고 메시지를 숨깁니다.
warnings.filterwarnings("ignore")

# 프로젝트의 설정값을 불러옵니다.
import config

# 벡터 저장소를 불러오는 함수를 가져옵니다.
from rag_app.indexer import get_store


# ==================================================
# 환경 변수와 벡터 저장소 준비
# ==================================================

# 프로젝트의 .env 파일을 불러옵니다.
load_dotenv()

# 프로그램이 시작될 때 벡터 저장소를 한 번만 준비합니다.
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
# Retriever Node
# ==================================================
def retriever_node(state: dict) -> dict:
    """
    State에서 검색 조건을 읽고 문서를 검색합니다.

    검색 성공 여부와 실패 원인을 함께 반환합니다.
    검색 결과가 없어도 예외를 발생시키지 않습니다.
    """

    # query가 있으면 query를 사용합니다.
    query = state.get("query")

    # query가 없으면 question을 검색어로 사용합니다.
    if not query:
        query = state.get("question", "")

    # State에 top_k가 없으면 config의 기본값을 사용합니다.
    k = state.get("top_k")

    if not k:
        k = config.TOP_K

    # State에 검색 방식이 없으면 config의 기본값을 사용합니다.
    search_type = state.get("search_type")

    if not search_type:
        search_type = config.SEARCH_TYPE

    # State에 최소 점수 기준이 있으면 그 값을 사용합니다.
    min_score = state.get("min_score")

    # 최소 점수 기준이 없으면 config의 기본값을 사용합니다.
    if min_score is None:
        min_score = config.MIN_SCORE

    # 질문이 비어 있으면 검색하지 않고 결과를 반환합니다.
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
            k,
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

    # 검색 결과의 점수만 별도로 저장합니다.
    scores = []

    for document, score in search_results:
        # MMR은 점수가 없으므로 None을 저장합니다.
        if score is None:
            scores.append(None)
        else:
            scores.append(round(float(score), 3))

    # 통과한 문서를 저장할 리스트입니다.
    kept_documents = []

    # Similarity 검색은 점수를 기준으로 문서를 필터링합니다.
    if search_results[0][1] is not None:

        for document, score in search_results:
            # 최소 점수 이상인 문서만 사용합니다.
            if float(score) >= float(min_score):
                kept_documents.append(document)

        # 문서가 하나도 통과하지 못했는지 기록합니다.
        if kept_documents:
            fail_reason = ""
        else:
            fail_reason = "low_score"

    # MMR 검색은 점수가 없으므로 검색 결과를 그대로 사용합니다.
    else:
        for document, score in search_results:
            kept_documents.append(document)

        # MMR 검색은 점수 미달 여부를 판단하지 않습니다.
        fail_reason = ""

    # 검색 결과를 그래프의 State에 반환합니다.
    return {
        "documents": kept_documents,
        "scores": scores,
        "retrieval_ok": len(kept_documents) > 0,
        "fail_reason": fail_reason,
        "log": [
            f"검색 방식={search_type}, "
            f"k={k}, "
            f"{len(kept_documents)}/{len(search_results)}건 통과"
        ],
    }


# ==================================================
# Retriever Node 단독 테스트
# ==================================================
if __name__ == "__main__":

    # Retriever Node를 테스트할 질문 목록입니다.
    test_cases = [
        {
            "query": "환불은 며칠 이내인가요?",
        },
        {
            "query": "반품하고 싶은데 언제까지?",
        },
        {
            "query": "대표이사가 누구인가요?",
        },
        {
            "query": "",
        },
        {
            "query": "환불 규정",
            "top_k": 10,
        },
        {
            "query": "환불과 교환 차이",
            "search_type": "mmr",
        },
    ]

    # 테스트 질문을 하나씩 실행합니다.
    for test_case in test_cases:

        # Retriever Node를 실행합니다.
        result = retriever_node(test_case)

        # 빈 질문이면 화면에 표시할 문구를 바꿉니다.
        question = test_case.get("query", "")

        if not question:
            question = "(빈 질의)"

        # 질문을 출력합니다.
        print(f"\nQ: {question}")

        # 통과한 문서 개수와 검색 성공 여부를 출력합니다.
        print(
            f"   통과 {len(result['documents'])}건 | "
            f"ok={result['retrieval_ok']} | "
            f"사유={result['fail_reason'] or '-'}"
        )

        # 검색 점수를 출력합니다.
        print(
            f"   점수: {result['scores'][:5]}"
        )