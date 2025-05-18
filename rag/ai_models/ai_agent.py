import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

client = Ollama(model="gwen3:14b")

chunk_size = 500
chunk_overlap = 0.25

file_path = "documents/Hardware.txt"
persist_directory = "chroma"

embedding = OllamaEmbeddings(model="bge-m3")

with open(file_path, "r", encoding="utf-8") as file:
            contents = file.read()

filename = os.path.basename(file_path)
metadatas = [{"nome do arquivo": filename}]


text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size,
                                        chunk_overlap=int(chunk_size * chunk_overlap),
                                        length_function=len,
                                        is_separator_regex=False,
                                        )

all_splits = text_splitter.create_documents([contents], metadatas=metadatas)

vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_directory)

question = "O que você sabe externo ao documento que você leu?"

docs = vectorstore.similarity_search_with_score(question, k=4)