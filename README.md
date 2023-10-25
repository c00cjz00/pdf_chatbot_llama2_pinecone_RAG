# pdf_chatbot_llama2_pinecone_RAG
### 0. 下載程式碼
```
git clone https://github.com/c00cjz00/pdf_chatbot_llama2_pinecone_RAG.git
```

### 1. 利用singularity啟動程式
```
cd pdf_chatbot_llama2_pinecone_RAG
ml libs/singularity/3.10.2
singularity exec --nv -B /work /work/u00cjz00/nvidia/pytorch_2.0.1-cuda11.7-cudnn8-runtime.sif pip3 install -r requirements.txt
singularity exec --nv -B /work /work/u00cjz00/nvidia/pytorch_2.0.1-cuda11.7-cudnn8-runtime.sif python3 ingest.py
singularity exec --nv -B /work /work/u00cjz00/nvidia/pytorch_2.0.1-cuda11.7-cudnn8-runtime.sif python3 model.py
```


### 3. 問題範例
```
# 針灸
ware is Acupuncture?
Give me some BOOKS about Acupuncture.

# 過敏
What are Allergies?

# 穴位
請列出跟肺臟有關的穴位
```
