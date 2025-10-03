import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_community.document_loaders import S3FileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp

# For embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings.embeddings import Embeddings  # interface
from typing import List

load_dotenv()

# Custom embedding class wrapper if needed
class HuggingFaceEmbeddingsLocal(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query], show_progress_bar=False)[0].tolist()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


S3_BUCKET = os.getenv("S3_BUCKET", "mental-chatbot-docs")
MODEL_PATH = os.getenv("LLAMA_CPP_MODEL_PATH", "models/qwen1_5-1_8b-chat-q4_k_m.gguf")
N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.0"))
N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
VERBOSE = os.getenv("LLAMA_VERBOSE", "false").lower() in ("true", "1", "yes")

persist_dir = "./chroma_store"
vectorstore = None

@app.on_event("startup")
def startup_load_docs():
    global vectorstore

    # Load PDF document from S3 (requires AWS credentials)
    loader = S3FileLoader(bucket=S3_BUCKET, key="cbt.pdf")
    docs = loader.load()

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Use local embedding
    embeddings_local = HuggingFaceEmbeddingsLocal()

    # Build vectorstore correctly
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings_local,
        persist_directory=persist_dir,
        collection_name="chatbot_docs"
    )

@app.get("/chat")
def chat(query: str = Query(...)):
    global vectorstore
    if vectorstore is None:
        return {"error": "Vector store not initialized"}

    retriever = vectorstore.as_retriever()

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        temperature=TEMPERATURE,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=VERBOSE
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa.run(query)
    return {"query": query, "answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
