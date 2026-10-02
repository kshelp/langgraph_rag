# ============================================================
# 30차시 - Rewrite Node
# 검색에 실패한 질문을 검색하기 좋은 표현으로 다시 작성합니다.
# 원본 question은 유지하고, 검색용 query만 변경합니다.
# ============================================================

import os
import sys
import warnings


# ------------------------------------------------------------
# 필요한 모듈의 경로를 설정합니다.
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

warnings.filterwarnings("ignore")

# 공통 모듈이 있는 src 폴더
sys.path.insert(0, SRC_DIR)

# 이전 차시에서 만든 모듈
sys.path.append(os.path.join(SRC_DIR, "13"))
sys.path.append(os.path.join(SRC_DIR, "18"))
sys.path.append(os.path.join(SRC_DIR, "24"))
sys.path.append(os.path.join(SRC_DIR, "25"))

# 현재 30차시 모듈을 가장 먼저 찾도록 합니다.
sys.path.insert(0, CURRENT_DIR)


# ------------------------------------------------------------
# 필요한 모듈을 가져옵니다.
# ------------------------------------------------------------

import config

from prompts import REWRITE_PROMPT
from generator import _llm
from langchain_core.output_parsers import StrOutputParser


# ------------------------------------------------------------
# Rewrite Chain을 만듭니다.
#
# Prompt → LLM → 문자열
# ------------------------------------------------------------

rewrite_chain = REWRITE_PROMPT | _llm | StrOutputParser()


# 최대 재작성 횟수입니다.
MAX_REWRITE = getattr(config, "MAX_REWRITE", 2)


# ============================================================
# Rewrite Node
# 검색에 실패한 질문을 새로운 검색어로 바꿉니다.
# 직접 검색하지 않고 query만 새로 만듭니다.
# ============================================================

def rewrite_node(state) -> dict:

    # 원본 사용자 질문을 가져옵니다.
    question = state.get("question", "")

    # 지금까지 사용한 검색어를 가져옵니다.
    tried = state.get("tried_queries", [])

    # 현재까지 재작성한 횟수를 가져옵니다.
    rewrites = state.get("rewrites", 0)

    # 검색 또는 검증 단계의 실패 이유를 가져옵니다.
    reason = state.get("reason")

    if not reason:
        reason = state.get("fail_reason")

    if not reason:
        reason = "관련 자료를 찾지 못함"


    # --------------------------------------------------------
    # 최대 재작성 횟수에 도달하면 중단합니다.
    # --------------------------------------------------------

    if rewrites >= MAX_REWRITE:

        return {
            "rewrite_ok": False,
            "log": [f"재작성 상한({MAX_REWRITE}) 도달"],
        }


    # --------------------------------------------------------
    # 이미 사용한 검색어 목록을 문자열로 만듭니다.
    # --------------------------------------------------------

    if tried:
        tried_text = ", ".join(tried)
    else:
        tried_text = "(없음)"


    # --------------------------------------------------------
    # LLM에게 새로운 검색어를 요청합니다.
    # --------------------------------------------------------

    try:

        new_query = rewrite_chain.invoke({
            "question": question,
            "tried": tried_text,
            "reason": reason,
        })

        # 앞뒤 공백과 따옴표를 제거합니다.
        new_query = new_query.strip()
        new_query = new_query.strip('"')
        new_query = new_query.strip("'")


    # LLM 호출 중 오류가 발생한 경우입니다.
    except Exception as e:

        return {
            "rewrite_ok": False,
            "rewrites": rewrites + 1,
            "log": [f"재작성 오류({type(e).__name__})"],
        }


    # --------------------------------------------------------
    # 새로운 검색어가 비어 있으면 실패입니다.
    # --------------------------------------------------------

    if not new_query:

        return {
            "rewrite_ok": False,
            "rewrites": rewrites + 1,
            "log": ["재작성 실패(빈값)"],
        }


    # --------------------------------------------------------
    # 이미 사용했던 검색어라면 다시 검색하지 않습니다.
    # --------------------------------------------------------

    if new_query in tried:

        return {
            "rewrite_ok": False,
            "rewrites": rewrites + 1,
            "log": [f"재작성 실패(중복): {new_query}"],
        }


    # --------------------------------------------------------
    # 새로운 검색어 생성에 성공했습니다.
    # --------------------------------------------------------

    return {
        # Retriever가 다음 검색에서 사용할 검색어입니다.
        "query": new_query,

        # 재작성 횟수를 1 증가시킵니다.
        "rewrites": rewrites + 1,

        # 새로운 검색어를 검색 이력에 추가합니다.
        "tried_queries": [new_query],

        # 새로운 query는 아직 검색하지 않은 상태입니다.
        "retrieval_ok": False,

        # Rewrite 성공 여부입니다.
        "rewrite_ok": True,

        # 실행 기록을 남깁니다.
        "log": [
            f"재작성 {rewrites + 1}회: "
            f"'{question[:20]}...' -> '{new_query}'"
        ],
    }