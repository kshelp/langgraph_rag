import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import streamlit as st

import config
from rag import ask


# ============================================================
# data 폴더(DOC_DIR)의 PDF를 브라우저에서 열 수 있게 제공합니다.
# 브라우저는 http 페이지에서 file:// 링크를 막기 때문에
# 별도 포트(DOC_PORT)로 읽기 전용 파일 서버를 띄웁니다.
# 이 PC(127.0.0.1)에서만 접속할 수 있습니다.
# ============================================================

class _QuietHandler(SimpleHTTPRequestHandler):

    # 요청마다 콘솔에 로그가 찍히지 않게 합니다.
    def log_message(self, format, *args):
        pass


@st.cache_resource
def _start_doc_server():

    handler = functools.partial(_QuietHandler, directory=str(config.DOC_DIR))

    try:
        server = ThreadingHTTPServer(("127.0.0.1", config.DOC_PORT), handler)
    except OSError:
        # 이미 다른 web.py 프로세스가 같은 포트로 제공 중입니다.
        return None

    threading.Thread(target=server.serve_forever, daemon=True).start()

    return server


_start_doc_server()


# 상대 경로("law/xxx.pdf")를 파일 서버 링크로 바꿉니다.
def _doc_link(path, page):
    return f"http://127.0.0.1:{config.DOC_PORT}/{quote(path)}#page={page}"

st.set_page_config(
    page_title="문서 Q&A",
    page_icon="■"
)

st.title("■ 문서 기반 질의응답")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input(
    "궁금한 것을 물어보세요"
)

if question:

    # 답변을 만드는 동안 로딩 메시지를 표시합니다.
    with st.spinner("자료를 찾는 중..."):

        # RAG 시스템에 질문을 전달합니다.
        result = ask(question)

    # 질문과 결과를 대화 기록에 저장합니다.
    st.session_state.history.append(
        (question, result)
    )

for question, result in st.session_state.history:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write(result["answer"])

        # 캐시에서 가져온 답변이면 표시합니다.
        if result.get("cached"):
            st.caption(f"⚡ 캐시된 답변 (유사도 {result['cache_score']:.3f})")

        # /recache로 새로 답해 캐시를 갱신했으면 표시합니다.
        if result.get("recached"):
            st.caption("🔄 새 답변으로 캐시를 갱신했습니다")

        if result["sources"]:

            # 참고 문서를 (파일, 페이지)마다 한 줄씩 링크로 만듭니다.
            lines = []
            seen = set()

            for source in result["sources"]:

                key = (source["file"], source["page"])

                if key in seen:
                    continue

                seen.add(key)

                # 파일명의 [ ]가 마크다운 링크 문법과 겹치지 않게 처리합니다.
                label = (
                    f"{source['file']} p.{source['page']}"
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                )

                # 원본 파일이 있으면 링크로, 없으면 이름만 표시합니다.
                if source.get("path"):
                    link = _doc_link(source["path"], source["page"])
                    lines.append(f"- [{label}]({link})")
                else:
                    lines.append(f"- {label}")

            # 참고 문서 목록을 출력합니다.
            st.caption("※ 참고")
            st.markdown("\n".join(lines))