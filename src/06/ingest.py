from langchain_community.document_loaders import PyPDFLoader
import os, re

NOISE = ["대외비"]

def load_documents(path):
    loader = PyPDFLoader(path)        
    docs = loader.load()

    for d in docs:
        text = d.page_content
        for noise in NOISE:
            text = text.replace(noise, "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        d.page_content = text.strip()

        # 메타데이터
        d.metadata["filename"] = os.path.basename(path)
        d.metadata["page_no"] = d.metadata.get("page", 0) + 1

    empty_pages = []
    
    for d in docs:
        text = d.page_content.strip()

        if len(text) < 10:
            page = d.metadata.get("page", "?")
            empty_pages.append(page)

    if empty_pages:
        print("스캔 pdf 일 수 있습니다.")

    total_len = 0
    for d in docs:
        total_len += len(d.page_content)
    print(f"{len(docs)}쪽 로딩 완료 (총 {total_len}자)")

    return docs
 

if __name__ == "__main__":
    docs = load_documents("../../data/manual.pdf")
    print(docs[0].page_content)