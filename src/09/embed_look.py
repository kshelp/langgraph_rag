from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

emb = OpenAIEmbeddings(model="text-embedding-3-small")

vec = emb.embed_query("환불 규정이 궁금해요")

print("차원 수 : ", len(vec))

first_10 = []

for value in vec[:10]:
    first_10.append(round(value, 4))

print("10개 : ", first_10)