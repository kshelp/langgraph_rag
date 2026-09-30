from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "../../data/notice.pdf",
    # encoding="utf-8"
)

documents = loader.load()

print(f"{len(documents)}");
print(f"{documents[0].page_content}");
print(f"{documents[0].metadata}");