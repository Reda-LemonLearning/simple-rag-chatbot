import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import os
from dotenv import load_dotenv
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext

def main() : 
    reader = SimpleDirectoryReader(input_dir="./data/pdf_folder")
    docs = reader.load_data()

    load_dotenv()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"

    SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    - Generate human readable output, avoid creating output with gibberish text.
    - Generate only the requested output, don't include any other language before or after the requested output.
    - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    - Generate professional language typically used in business documents in North America.
    - Never generate offensive or foul language.
    """

    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )

    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model_name,
        model_name=model_name,
        device_map="auto",
    )

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    embeddings = embed_model.get_text_embedding(
    "Open AI new Embeddings models is great."
)
    print(embeddings)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(docs)

    query_engine = index.as_query_engine()

    response = query_engine.query("What is the size of the sun ?")

    print(str(response))
    print(response.source_nodes[0].get_content()) 


if __name__ == "__main__":
    main()