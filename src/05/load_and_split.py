from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    "../../data/notice.txt",
    encoding="utf-8"
    )

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 30
)

chunks = splitter.split_documents(documents)

print(f"{len(chunks)}")
print(f"{chunks[0].page_content}")
print(f"{chunks[1].page_content}")