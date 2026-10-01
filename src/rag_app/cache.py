# ============================================================
# 질문-답변 시맨틱 캐시 (Qdrant)
# 질문을 임베딩해 이전에 답한 질문 중 충분히 비슷한 것이 있으면
# LLM을 다시 호출하지 않고 저장된 답변을 돌려줍니다.
#
# - 유사도 CACHE_MIN_SCORE 이상만 캐시로 인정합니다.
#   (실측: "비정상이면" vs "정상이면" 처럼 뜻이 반대인 질문도 0.73이 나와
#    낮은 기준은 엉뚱한 답을 돌려줄 수 있습니다.)
# - 모델·프롬프트·컬렉션이 바뀌면 VERSION이 달라져 이전 캐시를 쓰지 않습니다.
# - CACHE_TTL_DAYS가 지난 답변은 쓰지 않습니다.
# - 문서 인덱스를 다시 만들면 indexer.py가 캐시를 비웁니다(clear).
# - 질문 앞에 "/recache"를 붙이면 그 질문의 캐시를 지우고 새로 답합니다(invalidate).
# ============================================================

import re
import time
import unicodedata
import uuid

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models

import config


# 이 값이 다른 캐시는 조회하지 않습니다.
VERSION = "|".join([
    config.COLLECTION,
    config.EMBED_MODEL,
    config.LLM_MODEL,
    str(config.TEMPERATURE),
    config.PROMPT_VER,
])

_client = QdrantClient(url=config.QDRANT_URL)
_embeddings = None


# 보이지 않는 문자(BOM, 폭 없는 공백 등)와 공백 차이로
# 같은 질문의 유사도가 떨어지지 않게 정규화합니다.
# (실측: 앞에 BOM 하나만 붙어도 같은 질문이 1.00 → 0.96)
def _normalize(question):

    text = unicodedata.normalize("NFKC", question)
    text = re.sub(r"[﻿​-‍⁠]", "", text)

    return re.sub(r"\s+", " ", text).strip()


# 임베딩 모델은 처음 쓸 때 만듭니다(.env 로드 이후).
def _embed(question):

    global _embeddings

    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)

    return _embeddings.embed_query(_normalize(question))


# ============================================================
# 캐시 조회
# 반환: (캐시된 결과 또는 None, 질문 벡터)
# 질문 벡터는 저장(put)할 때 다시 임베딩하지 않도록 함께 돌려줍니다.
# ============================================================
def lookup(question):

    vector = _embed(question)

    if not _client.collection_exists(config.CACHE_COLLECTION):
        return None, vector

    # 같은 VERSION이고 TTL 안에 저장된 답변만 찾습니다.
    since = time.time() - config.CACHE_TTL_DAYS * 86400

    hits = _client.query_points(
        collection_name=config.CACHE_COLLECTION,
        query=vector,
        limit=1,
        with_payload=True,
        query_filter=models.Filter(must=[
            models.FieldCondition(
                key="version",
                match=models.MatchValue(value=VERSION)
            ),
            models.FieldCondition(
                key="created_at",
                range=models.Range(gte=since)
            ),
        ]),
    ).points

    if not hits or hits[0].score < config.CACHE_MIN_SCORE:
        return None, vector

    payload = hits[0].payload

    result = {
        "answer": payload["answer"],
        "sources": payload["sources"],
        "cited": payload["cited"],
        "ok": True,
        "cached": True,
        "cache_score": hits[0].score,
        "cache_question": payload["question"],
    }

    return result, vector


# ============================================================
# 캐시 저장
# ============================================================
def put(question, vector, result):

    # 컬렉션이 없으면 벡터 차원에 맞춰 만듭니다.
    if not _client.collection_exists(config.CACHE_COLLECTION):
        _client.create_collection(
            collection_name=config.CACHE_COLLECTION,
            vectors_config=models.VectorParams(
                size=len(vector),
                distance=models.Distance.COSINE
            ),
        )

    _client.upsert(
        collection_name=config.CACHE_COLLECTION,
        points=[models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "question": _normalize(question),
                "answer": result["answer"],
                "sources": result["sources"],
                "cited": result["cited"],
                "version": VERSION,
                "created_at": time.time(),
            },
        )],
    )


# ============================================================
# 특정 질문의 캐시 삭제 ("/recache" 요청 시 호출)
# 같은 질문으로 보는 답변(유사도 CACHE_MIN_SCORE 이상)을
# 버전·기간과 관계없이 모두 지웁니다.
# 반환: (질문 벡터, 삭제한 개수)
# ============================================================
def invalidate(question):

    vector = _embed(question)

    if not _client.collection_exists(config.CACHE_COLLECTION):
        return vector, 0

    hits = _client.query_points(
        collection_name=config.CACHE_COLLECTION,
        query=vector,
        limit=100,
        score_threshold=config.CACHE_MIN_SCORE,
    ).points

    if hits:
        _client.delete(
            collection_name=config.CACHE_COLLECTION,
            points_selector=models.PointIdsList(
                points=[hit.id for hit in hits]
            ),
        )

    return vector, len(hits)


# ============================================================
# 캐시 전체 삭제 (문서 인덱스를 다시 만들 때 호출)
# ============================================================
def clear():

    if _client.collection_exists(config.CACHE_COLLECTION):
        _client.delete_collection(config.CACHE_COLLECTION)
        print(f"✓ 답변 캐시 삭제: {config.CACHE_COLLECTION}")
