from pymongo import MongoClient
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
import os
import numpy as np

class VectorMongoDB:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client["financial_rag"]
        self.collection = self.db["documents"]
        self.embedder = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        
        self._initialize_index()

    def _initialize_index(self):
        if "vector_index" not in self.collection.list_search_indexes():
            self.collection.create_search_index([
                {
                    "fields": [{
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 768,
                        "similarity": "cosine"
                    }]
                }
            ])

    def add_documents(self, chunks: list):
        embeddings = list(self.embedder.embed(chunks))
        docs = [{
            "text": chunk,
            "embedding": [float(x) for x in embedding],
            "type": "document"
        } for chunk, embedding in zip(chunks, embeddings)]
        
        self.collection.insert_many(docs)

    def add_analysis(self, analysis_text: str):
        chunks = self._chunk_text(analysis_text)
        embeddings = list(self.embedder.embed(chunks))
        docs = [{
            "text": chunk,
            "embedding": [float(x) for x in embedding],
            "type": "analysis"
        } for chunk, embedding in zip(chunks, embeddings)]
        
        self.collection.insert_many(docs)

    def query_rag(self, query: str, context: str = "", k: int = 15):
        # Embed query
        query_embedding = [float(x) for x in next(self.embedder.embed([query]))]
        
        # Vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": k
                }
            },
            {"$project": {"text": 1, "_id": 0, "type": 1}}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        # Reranking con contexto
        passages = [
            {"id": i, "text": f"{doc['text']}\n{context}", "metadata": doc}
            for i, doc in enumerate(results)
        ]
        
        reranked = self.ranker.rerank(RerankRequest(
            query=query,
            passages=passages
        ))
        
        return "\n".join([r["text"] for r in reranked[:5]])

    @staticmethod
    def _chunk_text(text: str, chunk_size=1024, overlap=128):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks