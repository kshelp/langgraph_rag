from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "../../data/notice.pdf",
    # encoding="utf-8"
)

docs = loader.load()

def validate(docs):
    empty_page = []

    for d in docs:
        text = d.page_content.strip()
        if len(text) < 10:
            page = d.metadata.get("page", "?")
            empty_page.append(page)

    if empty_page: 
        print("스캔 pdf 일 수 있습니다.")

validate(docs)