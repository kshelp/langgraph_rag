from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "../../data/notice.txt",
    encoding="utf-8"
)

documents = loader.load()

print(f"{len(documents)}");
print(f"{documents[0].page_content}");
print(f"{documents[0].metadata}");