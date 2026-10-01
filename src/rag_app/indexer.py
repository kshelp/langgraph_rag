# ============================================================
# data/allganize_rag_ko/<domain>/*.pdf 를 읽어 청킹한 뒤
# Qdrant 컬렉션(config.COLLECTION)에 적재합니다.
#
# 실행: python indexer.py          (컬렉션 재생성)
#       python indexer.py --keep   (있으면 그대로 사용)
# 사용: from indexer import get_store
#       store.similarity_search(q, k=5, filter={"domain": "finance"})
# ============================================================

import re
import sys

import pymupdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

import cache
import config

DOMAINS = ["finance", "public", "medical", "law", "commerce"]

# 한 번에 임베딩할 청크 수 (OpenAI 요청 크기 제한 대비)
BATCH_SIZE = 500


# ============================================================
# PDF 문서를 읽고 텍스트와 메타데이터를 정리합니다.
# ============================================================
# 일부 PDF는 공백·글머리 기호를 제어문자(\x01, \x07 등)로 씁니다.
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# 정상 문서에 나오는 문자 블록: ASCII·Latin-1, 한글(자모·음절), 한자, 가나,
# 문장부호·기호·원문자·화살표, 전각 문자
NORMAL_CHARS = re.compile(
    r"[ -~ -ÿᄀ-ᇿ -⋿①-➿"
    r"　-ヿ㄰-㆏㈀-㋿一-鿿가-힣"
    r"豈-﫿＀-￯]")

# 폰트에 글자 매핑(ToUnicode)이 없는 PDF는 구르무키·키릴·콥트 문자 등으로 깨집니다.
# 실측: 깨진 페이지 ≤ 0.77, 정상 페이지 ≥ 0.94
MIN_NORMAL_RATIO = 0.85


def _clean(text):
    text = CONTROL.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 연속된 빈 줄을 최대 두 줄로 정리합니다.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_garbled(text):
    """폰트 매핑이 깨진 페이지인지 판단합니다."""
    chars = re.sub(r"\s", "", text)
    return len(NORMAL_CHARS.findall(chars)) / max(len(chars), 1) < MIN_NORMAL_RATIO


def _load():
    docs = []
    skipped = {}   # source → 제외된 페이지 수 (OCR 대상)

    for domain in DOMAINS:
        for path in sorted((config.DOC_DIR / domain).glob("*.pdf")):
            source = f"{domain}/{path.name}"

            # pypdf는 일부 한글 CID 폰트(/UniKS-UTF16-H 등)를 읽지 못해 PyMuPDF를 씁니다.
            try:
                pdf = pymupdf.open(path)
            except Exception as error:
                print(f"  ✗ 읽기 실패: {source} ({error})")
                continue

            for page_index, page in enumerate(pdf):
                text = _clean(page.get_text())

                # 텍스트가 없는 페이지(스캔 이미지)나 깨진 페이지는 제외합니다.
                if not text or _is_garbled(text):
                    skipped[source] = skipped.get(source, 0) + 1
                    continue

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "domain": domain,
                        "filename": path.name,
                        "source": source,
                        "page_no": page_index + 1,
                    },
                ))
            pdf.close()

    for source, count in skipped.items():
        print(f"  ⚠ 텍스트 없음/깨짐 {count}쪽 제외 (OCR 필요): {source}")

    return docs


# ============================================================
# 페이지를 검색에 적합한 작은 조각(chunk)으로 나눕니다.
# ============================================================
def _split(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


# ============================================================
# Qdrant 벡터 저장소를 준비합니다.
# 기존 컬렉션이 있으면 연결하고, 없으면 새로 생성합니다.
# ============================================================
def get_store(rebuild=False):
    load_dotenv()
    embeddings = OpenAIEmbeddings(model=config.EMBED_MODEL)
    client = QdrantClient(url=config.QDRANT_URL)

    if client.collection_exists(config.COLLECTION) and not rebuild:
        return QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION,
            embedding=embeddings
        )

    docs = _load()
    chunks = _split(docs)
    if not chunks:
        raise RuntimeError(f"적재할 문서가 없습니다: {config.DOC_DIR}")

    files = {d.metadata["source"] for d in docs}
    print(f"문서 {len(files)}개, 페이지 {len(docs)}개 → 청크 {len(chunks)}개")

    # 배치로 나누어 임베딩하고 하나의 컬렉션에 적재합니다.
    # 첫 배치에서 기존 컬렉션을 지우고 새로 만듭니다.
    store = None
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        if store is None:
            store = QdrantVectorStore.from_documents(
                batch,
                embeddings,
                url=config.QDRANT_URL,
                collection_name=config.COLLECTION,
                force_recreate=True
            )
        else:
            store.add_documents(batch)
        print(f"  임베딩 {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"✓ Qdrant 적재 완료: {config.COLLECTION}")

    # 문서가 바뀌었으니 이전 답변 캐시는 지웁니다.
    cache.clear()

    return store


if __name__ == "__main__":
    get_store(rebuild="--keep" not in sys.argv)
