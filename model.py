## 問答

# 01: Configure
http_proxy='http://lgn304-v304:53128'
PINECONE_API_KEY='20163887-a4fa-44e7-98d2-ab1eb38937f6'
PINECONE_API_ENV='gcp-starter'
index_name="cjz-medical"
Embeddings_ID="sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID="/work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf"

# 02: Load LIBRARY
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain

import transformers
import torch
import pinecone

# 03: Embeddings 模型 384維度 
embeddings=HuggingFaceEmbeddings(model_name=Embeddings_ID)

# 04: LLM模型
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
pipeline=transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1024,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
    )
llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
    
# 05: 連線 pinecone 向量資料庫
openapi_config = OpenApiConfiguration.get_default_copy()
openapi_config.proxy = http_proxy
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV,openapi_config=openapi_config)

# 06: 搜尋 pinecone 向量資料庫, 列出前三名
docsearch=Pinecone.from_existing_index(index_name, embeddings)
query = "What are Allergies"
docs=docsearch.similarity_search(query, k=3)

# 07. LLM彙整
chain = load_qa_chain(llm, chain_type="stuff")
result=chain.run(input_documents=docs, question=query)
print(result)