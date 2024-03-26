from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

reader = SimpleDirectoryReader(input_dir="./data/pdf_folder")
docs = reader.load_data()

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llama = Ollama(model="llama2", request_timeout=40.0)

Settings.llm = llama
Settings.embed_model = embedding_model

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

question = "What is the size of the sun ?"

print(query_engine.query(question))