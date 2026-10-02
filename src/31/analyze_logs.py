# ============================================================
# RAG 실행 로그를 읽어 간단한 통계를 출력합니다.
# ============================================================

import json
from collections import Counter


# 로그 읽기
with open("logs/queries.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]


# 로그가 없으면 종료
if not rows:
    print("로그가 없습니다.")
    raise SystemExit


# 기본 통계
print("총 질문:", len(rows))

pass_count = sum(
    1 for r in rows
    if r.get("grade") == "pass"
)

print(
    "검증 통과율:",
    f"{pass_count / len(rows) * 100:.0f}%"
)

print(
    "유형 분포:",
    Counter(r.get("intent", "") for r in rows)
)


# 재작성 및 오류 횟수
rewrite_count = sum(
    1 for r in rows
    if r.get("rewrites", 0) > 0
)

error_count = sum(
    1 for r in rows
    if r.get("node_error", "")
)


# 결과 출력
print(
    "재작성 발생률:",
    f"{rewrite_count / len(rows) * 100:.0f}%"
)

print(
    "평균 응답:",
    f"{sum(r.get('elapsed', 0) for r in rows) / len(rows):.1f}초"
)

print("오류 발생:", error_count)