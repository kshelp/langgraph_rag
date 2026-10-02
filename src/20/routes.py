# ============================================================
# 검색 결과에 따라 다음 실행 경로를 결정합니다.
# 검색 성공이면 "ok", 검색 실패이면 "empty"를 반환합니다.
# ============================================================


def route_after_retrieve(state: dict) -> str:
    # 검색 성공 여부를 가져옵니다.
    retrieval_ok = state.get("retrieval_ok")

    # 검색에 성공한 경우
    if retrieval_ok:
        return "ok"

    # 검색에 실패한 경우
    return "empty"
