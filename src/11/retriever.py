import os
import sys
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '10'))
from indexer import get_store

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

# ========== 설정 ==========
TOP_K      = 3
MIN_SCORE  = 0.1
SEARCH_TYPE = "mmr"  # similarity / mmr

_store = get_store()

# ========== Part 1. 기본 함수 ==========
def show(name, retriever):
    """Retriever 결과 출력"""
    question = "환불과 교환은 어떻게 다른가요?"
    docs = retriever.invoke(question)
    print(f"■ {name}  ({len(docs)}개)")
    for d in docs:
        preview = d.page_content[:55].replace("\n", " ")         
        print(f"   p.{d.metadata['page_no']}  {preview}...")     
    print()

def compare_k(question, ks=(1, 3, 5)):
    """k값에 따른 점수 비교"""
    print(f"질문: {question}")
    print("=" * 58)
    for k in ks:
        pairs = _store.similarity_search_with_relevance_scores(question, k=k)
        scores = [s for _, s in pairs]
        low = sum(1 for s in scores if s < MIN_SCORE)
        print(f"k={k:<3} 점수={[round(float(s), 3) for s in scores]}")
        print(f"     기준 미달({MIN_SCORE}) 조각: {low}개")
    print()

# ========== Part 2. 심화 함수 ==========
def get_retriever(k=TOP_K, search_type=SEARCH_TYPE, category=None):
    """검색 옵션에 따라 Retriever 생성"""
    kwargs = {"k": k}
    if search_type == "mmr":
        kwargs.update({"fetch_k": k * 4, "lambda_mult": 0.5})
    if category:
        kwargs["filter"] = {"category": category}
    return _store.as_retriever(
        search_type=search_type, 
        search_kwargs=kwargs
    )

def search(question, k=TOP_K, category=None):
    """점수 필터를 포함한 검색"""
    if category:
        pairs = _store.similarity_search_with_relevance_scores(
            question, k=k, filter={"category": category}
        )
    else:
        pairs = _store.similarity_search_with_relevance_scores(question, k=k)
    
    # 점수 기준 이상인 문서만 반환
    good = []
    for d, s in pairs:
        if s >= MIN_SCORE:
            good.append(d)
    return good

def build_context(docs):
    """검색 결과를 읽기 좋게 포매팅"""
    parts = []
    for i, d in enumerate(docs, 1):
        head = (f"[{i}] {d.metadata.get('filename','?')} "
                f"p.{d.metadata.get('page_no','?')}")
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

# ========== 테스트 ==========
if __name__ == "__main__":
    # Part 1. 기본 개념
    print("=" * 58)
    print("Part 1. Retriever 기본 개념")
    print("=" * 58 + "\n")
    
    show("similarity k=3",
         _store.as_retriever(search_kwargs={"k": 3}))
    
    show("MMR k=3 fetch_k=10",
         _store.as_retriever(search_type="mmr",
             search_kwargs={"k": 3, "fetch_k": 10}))
    
    # k값 비교
    print("=" * 58)
    print("k값에 따른 유사도 점수")
    print("=" * 58 + "\n")
    
    compare_k("환불은 며칠 이내인가요?")
    compare_k("환불과 교환은 어떻게 다른가요?")
    
    # Part 2. 심화: 통합 함수
    print("=" * 58)
    print("Part 2. 통합 검색 함수")
    print("=" * 58 + "\n")
    
    docs = search("환불과 교환의 차이는?")
    print(f"검색 결과: {len(docs)}개\n")
    if docs:
        print(build_context(docs))
    else:
        print("기준 이상의 관련 문서가 없습니다.")