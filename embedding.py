import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class Embedding(Embeddings):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"▶ Embedding model loaded on device: {self.device}")

    def encode(self, text):
        embeddings = self.model.encode(text)
        return embeddings

    def embed_query(self, text):
        embedding = self.encode([text])[0]
        return embedding.tolist()
    
    def embed_documents(self, docs):
        embeddings = self.encode(docs)
        return embeddings.tolist()