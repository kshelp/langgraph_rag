from pathlib import Path
BASE = Path(__file__).resolve().parent


# ── 문서 처리 ──
DOC_PATH      = BASE.parent.parent / "data/manual.pdf"
DOC_DIR       = BASE / "data/allganize_rag_ko"    # allganize_download.py 저장 위치
DOC_PORT      = 8600                              # web.py가 DOC_DIR을 제공하는 파일 서버 포트
CHUNK_SIZE    = 500      # 500→300: +10%p, 토큰 33% 절감
CHUNK_OVERLAP = 50       # chunk_size의 10%
# ── 임베딩 · 저장소 ──
EMBED_MODEL   = "text-embedding-3-small"   # ⚠ 변경 시 인덱스 재생성 
QDRANT_URL    = "http://127.0.0.1:6333"   # WSL Docker Qdrant (localhost는 IPv6로 해석돼 지연됨)
COLLECTION    = "rag_app_allganize"       # 기존 allganize_rag_ko(1024차원)와 분리
# ── 검색 ──
TOP_K         = 5        # 3→5: +10%p (k=8은 노이즈로 하락) 
MIN_SCORE     = -0.01     # 9차시 실측값
SEARCH_TYPE   = "similarity"
# ── 생성 ──
LLM_MODEL     = "gpt-4o-mini"
TEMPERATURE   = 0
PROMPT_VER    = "v3"     # v2→v3: +10%p, 토큰 증가 없음


# ── 답변 캐시 ──
CACHE_ENABLED    = True
CACHE_COLLECTION = "rag_app_answer_cache"   # 기존 answer_cache(1024차원)와 분리
CACHE_MIN_SCORE  = 0.95     # 사실상 같은 질문만 (실측: 반대 뜻 질문 0.73, 표현 변경 0.71~0.82)
CACHE_TTL_DAYS   = 7
RECACHE_PREFIX   = "/recache"   # 질문 앞에 붙이면 캐시를 무시하고 새로 답해 다시 저장


# ── 메시지 ──
MSG_NO_DOC    = "관련 자료를 찾지 못했습니다."
MSG_ERROR     = "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."