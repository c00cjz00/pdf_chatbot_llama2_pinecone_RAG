## 知識庫製作

# 01: Configure
pdf_file='Medical_Chatbot.pdf'
http_proxy='http://lgn304-v304:53128'
PINECONE_API_KEY='20163887-a4fa-44e7-98d2-ab1eb38937f6'
PINECONE_API_ENV='gcp-starter'
index_name="cjz-medical"

# 02: Load LIBRARY
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
#from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
import pinecone

# 03: Locad PDF
loader= PyPDFLoader(pdf_file)
data=loader.load()

# 04: Text splitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs=text_splitter.split_documents(data)

# 05: Embeddings 模型 384維度 
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 06: 儲存至 pinecone 向量資料庫
openapi_config = OpenApiConfiguration.get_default_copy()
openapi_config.proxy = http_proxy
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV,openapi_config=openapi_config)
docsearch = Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)
