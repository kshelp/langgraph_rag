# ═════════════════════════════════════════════
#  RAG 설정
#  최종 결정 : 2026-08-03
#  근거      : 14차시 벤치마크 (질문 20개 기준)
#              기준선 60% → 최종 90% (+30%p)
# ═════════════════════════════════════════════
# ── 문서 처리 ──
DOC_PATH      = "../../data/manual.pdf"
CHUNK_SIZE    = 500      # 500→300: +10%p, 토큰 33% 절감
CHUNK_OVERLAP = 50       # chunk_size의 10%
# ── 임베딩 · 저장소 ──
EMBED_MODEL   = "text-embedding-3-small"   # ⚠ 변경 시 인덱스 재생성 
INDEX_PATH    = "faiss_index"
# ── 검색 ──
TOP_K         = 5        # 3→5: +10%p (k=8은 노이즈로 하락) 
MIN_SCORE     = 0.0     # 9차시 실측값
SEARCH_TYPE   = "similarity"
# ── 생성 ──
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
PROMPT_VER = "v3"
MAX_RETRY = 3
USE_LLM_CLASSIFY = True

# ── 메시지 ──
MSG_NO_DOC = "관련 자료를 찾지 못했습니다."
MSG_ERROR = "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."