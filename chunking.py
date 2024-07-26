from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

import json

transcript = json.load(open("labeled_surgical_videos/transcript/10.json"))["text"]

text_splitter = SemanticChunker(HuggingFaceEmbeddings())

docs = text_splitter.create_documents([transcript])

print(docs)



